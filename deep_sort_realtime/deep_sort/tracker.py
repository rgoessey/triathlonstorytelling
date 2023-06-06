# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from datetime import datetime
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from math import sqrt
from .track import Track
from .triatlete import Triatlete
from .track import TrackState
import math

#get an intersection between two bounding boxes
def get_iou(bb1, bb2):
    """assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]"""

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_dist(loc1,loc2):
    # Bereken het centrale punt van elke LTRB-locatie
    center1 = [(loc1[0] + loc1[2]) / 2, (loc1[1] + loc1[3]) / 2]
    center2 = [(loc2[0] + loc2[2]) / 2, (loc2[1] + loc2[3]) / 2]
    
    # Bereken de Euclidische afstand tussen de centrale punten
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance

# Get the distance of 2 bounding boxes
def getBboxDistance(bbox1,bbox2):
    return sqrt((bbox1[0] - bbox2[0])**2 + (bbox1[1] - bbox2[1])**2 + (bbox1[2] - bbox2[2])**2 + (bbox1[3] - bbox2[3])**2)

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    today: Optional[datetime.date]
            Provide today's date, for naming of tracks

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    gating_only_position : Optional[bool]
        Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
    end : float
        The end time for how long the video will take
    fps : int
        For setting the speed of the frames per second
    time : int
        count the frames that were processed
    numberTracks : dict
        All tracks that have numbers coupled to their track id
    colorTracks : dict
        All tracks that have no numbers coupled to their track id
    closeTracks : dict
        Tracks that have other tracks who could be the same object
    storytelling : Storytelling
        Attribute for doing the storytelling
    """

    def __init__(
        self,
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        override_track_class=None,
        today=None,
        gating_only_position=False,
        end=0
    ):
        self.today = today
        self.metric = metric
        self.fps=25
        self.end=end
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.gating_only_position = gating_only_position
        self.time=0
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.numberTracks={}
        self.colorTracks={}
        self.closeTracks={}
        self.del_tracks_ids = []
        self._next_id = 1
        if override_track_class:
            self.track_class = override_track_class
        else:
            self.track_class = Track

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    

    def update(self, detections, triatletes,today=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks
        """
        if self.today:
            if today is None:
                today = datetime.now().date()
            # Check if its a new day, then refresh idx
            if today != self.today:
                self.today = today
                self._next_id = 1

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx],triatletes[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(self.time,detections[detection_idx],triatletes[detection_idx])
        new_tracks = []
        self.del_tracks_ids = []
        for t in self.tracks:
            if not t.is_deleted():
                new_tracks.append(t)
            else:
                # To set the stoptime of a track
                t.stoptime=self.time
                self.del_tracks_ids.append(t.track_id)
            # Add to numbertracks if numbers were detected
            if t.triatlete.number != -1:
                self.numberTracks[t.track_id]=t
            else:
                self.colorTracks[t.track_id]=t    
        self.tracks = new_tracks
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = [track.features[-1]]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )
        #add 1 frame to processed frames
        for t1 in self.tracks:
            for t2 in self.tracks:
                if t1.track_id != t2.track_id:
                    if t1.time_since_update < 4 and t2.time_since_update<4 and (get_iou(t1.to_ltrb(),t2.to_ltrb())>0.0 or get_dist(t1.to_ltrb(),t2.to_ltrb()) < 100):
                        if t1.track_id in self.closeTracks:
                            if t2.track_id not in self.closeTracks[t1.track_id]:
                                self.closeTracks[t1.track_id].append(t2.track_id)
                        else:
                            self.closeTracks[t1.track_id]=[t2.track_id]
        self.time+=1


    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices, only_position=self.gating_only_position
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    

    def _initiate_track(self, time,detection,triatlete):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        #Zelf toegevoegd
        b=True
        """for index,t in enumerate(self.tracks):
            if not t.is_confirmed():
                continue
            if get_iou(detection.to_tlbr(),t.to_tlbr())>0.1 and t.time_since_update>0:
                self.tracks[index].update(self.kf,detection,triatlete)
                b=False
                break"""
        if b:
            if self.today:
                track_id = "{}_{}".format(self.today, self._next_id)
            else:
                track_id = "{}".format(self._next_id)
            self.tracks.append(
                self.track_class(
                    mean,
                    covariance,
                    track_id,
                    self.n_init,
                    self.max_age,
                    time,
                    triatlete=Triatlete(triatlete["name"],triatlete["prob"],triatlete["number"]),
                    # mean, covariance, self._next_id, self.n_init, self.max_age,
                    feature=detection.feature,
                    original_ltwh=detection.get_ltwh(),
                    det_class=detection.class_name,
                    det_conf=detection.confidence,
                    instance_mask=detection.instance_mask,
                    others=detection.others,
                )
            )
            self._next_id += 1

    def delete_all_tracks(self):
        self.tracks = []
        self._next_id = 1

    """#Make sure that in a new scene not the same tracks will appear
    def startNewScene(self):
        for t in self.tracks:
            if not t.is_deleted:
                t.state == TrackState.Deleted
                t.stoptime=self.time
                self.del_tracks_ids.append(t.track_id)"""

    # called by main to give storytelling of scene before
    # First updates lists of storytelling with the corresponding one from tracking
    # then calls the storytelling function to give triatlet + timeframe
    def geefStorytellingScene(self):
        return self.numberTracks,self.colorTracks,self.closeTracks
    
    # To change fps in the main
    def updateFps(self,fps):
        self.fps=fps