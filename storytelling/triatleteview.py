from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
class Triatleteview:
    """
    A triatlete can be picked to show all scenes of 

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    """

    def __init__(
        self,
        videoUrl="",
        videoToUrl="",
        triatleteNr=None,
        scenes={}
    ):
       self.scenes=scenes
       self.triatleteNr=triatleteNr
       self.timeFrames=[]
       self.videoUrl=videoUrl
       self.videoToUrl=videoToUrl
       self.sceneviews=[]

    def processTriatlete(self):
        for index,s in enumerate(self.scenes):
            for t in s:
                if t==self.triatleteNr:
                    self.timeFrames.append([s[t][1],s[t][2]])
                    ffmpeg_extract_subclip(self.videoUrl, s[t][1],s[t][2], targetname=self.videoToUrl+ "frame"+ str(index)+ s[t][3] +".mp4")
        return self.timeFrames
           


