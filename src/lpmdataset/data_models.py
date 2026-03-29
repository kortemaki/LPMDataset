from collections.abc import Iterator
from enum import Enum, StrEnum
from functools import cached_property
import json
import os

import pandas as pd
from pydantic import BaseModel, computed_field, ConfigDict

from lpmdataset.modalities.asr import ASR
from lpmdataset.modalities.ocr import OCR, load_ocr_data
from lpmdataset.modalities.mouse import MouseTrace, load_trace_data


DATA_DIR = os.environ['DATASET_DIR']
FIGURES_DIR = os.environ['FIGURES_DIR']
SLIDE_IMAGES_DIR = os.environ['SLIDE_IMAGES_DIR']


class SlideRegion(Enum):
    OTHER = 0
    OCR = 1
    IMAGE = 2
    DIAGRAM = 3
    TABLE = 4
    EQUATION = 5


class Folder(StrEnum):
    ANAT_1 = 'anat-1/AnatomyPhysiology'
    ANAT_2 = 'anat-2/unordered'
    BIO_1 = 'bio-1/unordered'
    BIO_3 = 'bio-3/Biol1020'
    BIO_4 = 'bio-4/unordered'
    DENTAL_ANATOMY = 'dental/HeadNeckAnatomyINBDE'
    DENTAL_CARIO = 'dental/Cariology'
    DENTAL_EVIDENCEBASED = 'dental/Evidence-BasedDentistry'
    DENTAL_ENDODONTICS = 'dental/EndodonticsNBDEPartII'
    DENTAL_MEDICINE = 'dental/OralMedicineINBDE'
    DENTAL_NBDE = 'dental/NBDEPartI'
    DENTAL_OCCLUSION = 'dental/Occlusion'
    DENTAL_OPERATIVE = 'dental/OperativeDentistryNBDEPartII'
    DENTAL_ORTHO = 'dental/Orthodontics'
    DENTAL_ORTHO2 = 'dental/OrthodonticsNBDEPartII'
    DENTAL_PATHOLOGY = 'dental/OralPathologyNBDEPartII'
    DENTIAL_PEDIATRIC = 'dental/PediatricDentistryNBDEPartII'
    DENTAL_PERIODONTICS = 'dental/PeriodonticsNBDEPartII'
    DENTAL_PROSTHO = 'dental/ProsthodonticsNBDEPartII'
    DENTAL_SURGERY = 'dental/OralSurgeryNBDEPartII'
    DENTAL_RADIOLOGY = 'dental/OralRadiologyNBDEPartII'
    DENTAL_BASICS = 'dental/TheBasicsofDentistry'
    DENTAL_PATIENTS = 'dental/PatientManagementNBDEPartII'
    DENTAL_PERCEPTUAL = 'dental/PerceptualAbilityTestDAT'
    ML_1 = 'ml-1/MultimodalMachineLearning'
    PSY_1 = 'psy-1/LectureSeriesForIntrotoPsy-PSY101'
    PSY_2 = 'psy-2/LectureSeriesforIntrotoDevelopmentalPsy'
    SPEAKING = 'speaking/EssayWritingandPresentationskills'


FOLDER_TO_FIGURES = {
    Folder.ANAT_1: 'anat-1',
    Folder.ANAT_2: 'anat-2',
    Folder.BIO_1: 'bio-1',
    Folder.BIO_3: 'bio-3',
    Folder.BIO_4: 'bio-4',
    Folder.DENTAL_ANATOMY: 'dental',
    Folder.DENTAL_CARIO: 'dental',
    Folder.DENTAL_EVIDENCEBASED: 'dental',
    Folder.DENTAL_ENDODONTICS: 'dental',
    Folder.DENTAL_MEDICINE: 'dental',
    Folder.DENTAL_NBDE: 'dental',
    Folder.DENTAL_OCCLUSION: 'dental',
    Folder.DENTAL_OPERATIVE: 'dental',
    Folder.DENTAL_ORTHO: 'dental',
    Folder.DENTAL_ORTHO2: 'dental',
    Folder.DENTAL_PATHOLOGY: 'dental',
    Folder.DENTIAL_PEDIATRIC: 'dental',
    Folder.DENTAL_PERIODONTICS: 'dental',
    Folder.DENTAL_PROSTHO: 'dental',
    Folder.DENTAL_SURGERY: 'dental',
    Folder.DENTAL_RADIOLOGY: 'dental',
    Folder.DENTAL_BASICS: 'dental',
    Folder.DENTAL_PATIENTS: 'dental',
    Folder.DENTAL_PERCEPTUAL: 'dental',
    Folder.ML_1: 'ml-1',
    Folder.PSY_1: 'psy-1',
    Folder.PSY_2: 'psy-2',
    Folder.SPEAKING: 'speaking',
}

class Resolution(Enum):
    R240P  = ("240p", 426, 240)
    R360P  = ("360p", 640, 360)
    R480P  = ("480p", 854, 480)
    R720P  = ("720p", 1280, 720)
    R1080P = ("1080p", 1920, 1080)
    R1440P = ("1440p", 2560, 1440)
    SXGA   = ("SXGA", 1280, 1024)

    def __init__(self, label: str, width: int, height: int):
        self._label = label
        self.width = width
        self.height = height

    @property
    def label(self) -> str:
        return self._label

    @property
    def tuple(self) -> tuple:
        return (self.width, self.height)

    def __str__(self) -> str:
        return self._label

figure_bbs = pd.read_csv(os.path.join(DATA_DIR, "figure_annotations.csv"))

SPEAKER_RESOLUTIONS = {
  'dental/OperativeDentistryNBDEPartII': Resolution.R720P,
  'dental/PediatricDentistryNBDEPartII': Resolution.R1080P,
  'dental/Evidence-BasedDentistry': Resolution.R720P,
  'dental/EndodonticsNBDEPartII': Resolution.R1080P,
  'psy-2/LectureSeriesforIntrotoDevelopmentalPsy': Resolution.R720P,
  'dental/OralMedicineINBDE': Resolution.R1080P,
  'dental/Orthodontics': Resolution.R720P,
  'dental/ProsthodonticsNBDEPartII': Resolution.R1080P,
  'dental/NBDEPartI': Resolution.R720P,
  'anat-1/AnatomyPhysiology': Resolution.R1080P,
  'psy-1/LectureSeriesForIntrotoPsy-PSY101': Resolution.R720P,
  'dental/OralSurgeryNBDEPartII': Resolution.R1080P,
  'dental/OralPathologyNBDEPartII': Resolution.R1080P,
  'bio-1/unordered': Resolution.R1080P,
  'speaking/EssayWritingandPresentationskills': Resolution.R480P,
  'dental/PeriodonticsNBDEPartII': Resolution.R1080P,
  'bio-4/unordered': Resolution.R720P,
  'dental/Cariology': Resolution.R480P,
  'dental/OralRadiologyNBDEPartII': Resolution.R720P,
  'dental/TheBasicsofDentistry': Resolution.R1080P,
  'dental/PatientManagementNBDEPartII': Resolution.R1080P,
  'bio-3/Biol1020': Resolution.R480P,
  'dental/PerceptualAbilityTestDAT': Resolution.R720P,
  'ml-1/MultimodalMachineLearning': Resolution.SXGA,
  'dental/OrthodonticsNBDEPartII': Resolution.R720P,
  'anat-2/unordered': Resolution.R480P,
  'dental/HeadNeckAnatomyINBDE': Resolution.R720P,
  'dental/Occlusion': Resolution.R480P,
}

class Presentation(BaseModel):
    yt_id: str
    folder: Folder

    @classmethod
    def from_directory(cls, path: str) -> "Presentation":
        """Construct a Presentation from a video directory path.

        Finds the ``{yt_id}_transcripts.csv`` file inside *path* to determine
        the YouTube ID and derives the :class:`Folder` from the parent
        directory of *path* relative to ``DATASET_DIR``.
        """
        path = os.path.normpath(path)
        yt_id = None
        for fname in os.listdir(path):
            if fname.endswith('_transcripts.csv'):
                yt_id = fname[:-len('_transcripts.csv')]
                break
        if yt_id is None:
            raise FileNotFoundError(
                f"No _transcripts.csv file found in {path}"
            )
        folder_str = os.path.relpath(
            os.path.dirname(path), DATA_DIR
        ).replace(os.sep, '/')
        return cls(yt_id=yt_id, folder=Folder(folder_str))

    @computed_field
    @cached_property
    def dir_path(self) -> str:
        for fname in os.listdir(os.path.join(DATA_DIR, self.folder)):
            d = os.path.join(DATA_DIR, self.folder, fname)
            if not os.path.isdir(d):
                continue
            if fname == self.yt_id or any(f.startswith(self.yt_id) for f in os.listdir(d)):
                return d

    @computed_field
    @cached_property
    def figure_path(self) -> str:
        return os.path.join(SLIDE_IMAGES_DIR, FOLDER_TO_FIGURES[self.folder], self.yt_id)

    def slides(self) -> Iterator["Slide"]:
        for i in range(1000):
            s = Slide(presentation=self, slide_no=i)
            if os.path.exists(s.png): # TODO and os.path.exists(s.)
                yield s


def iter_presentations(folder: Folder) -> Iterator[Presentation]:
    """Yield a :class:`Presentation` for every valid video directory in *folder*."""
    folder_path = os.path.join(DATA_DIR, folder)
    for name in sorted(os.listdir(folder_path)):
        subdir = os.path.join(folder_path, name)
        if not os.path.isdir(subdir):
            continue
        try:
            yield Presentation.from_directory(subdir)
        except FileNotFoundError:
            continue


def iter_slides(presentation: Presentation) -> Iterator["Slide"]:
    """Yield all :class:`Slide` objects for the given *presentation*."""
    yield from presentation.slides()


class Slide(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    presentation: Presentation
    slide_no: int

    @computed_field
    @cached_property
    def png(self) -> str:
        return os.path.join(
            self.presentation.figure_path,
            f"slide_{self.slide_no:03d}.png",
        )

    @computed_field
    @cached_property
    def asr_file(self) -> str:
        return os.path.join(
            self.presentation.dir_path,
            f"slide_{self.slide_no:03d}_spoken.csv",
        )

    @computed_field
    @cached_property
    def asr_text(self) -> ASR:
        return ASR(path=self.asr_file)

    @computed_field
    @cached_property
    def ocr_file(self) -> str:
        return os.path.join(
            self.presentation.dir_path,
            f"slide_{self.slide_no:03d}_ocr.csv",
        )

    @computed_field
    @cached_property
    def ocr_text(self) -> OCR:
        return OCR(df=load_ocr_data(self.ocr_file))

    @computed_field
    @cached_property
    def trace_file(self) -> str:
        return os.path.join(
            self.presentation.dir_path,
            f"slide_{self.slide_no:03d}_trace.csv",
        )

    @computed_field
    @cached_property
    def mouse_trace(self) -> MouseTrace:
        return MouseTrace(df=load_trace_data(self.trace_file))

    @computed_field
    @cached_property
    def figures(self) -> pd.DataFrame:
        """Return figure bounding boxes for this slide as a DataFrame."""
        key = f"data/{self.presentation.folder}/{self.presentation.yt_id}/slide_{self.slide_no:03d}.jpg"
        row = figure_bbs[figure_bbs['Input.save_dir'] == key]
        if row.empty:
            return pd.DataFrame(columns=["label", "left", "top", "height", "width"])
        bboxes = json.loads(row.iloc[0]['boundingBoxes'])
        return pd.DataFrame(bboxes, columns=["label", "left", "top", "height", "width"])

    _FIGURE_LABEL_TO_REGION = {
        "Image": SlideRegion.IMAGE,
        "Diagram": SlideRegion.DIAGRAM,
        "Table": SlideRegion.TABLE,
        "Equation": SlideRegion.EQUATION,
    }

    def get_region_for_point(self, x: float, y: float) -> SlideRegion:
        """Return the SlideRegion for the unnormalized point (x, y)."""
        # Check figure bounding boxes first
        figs = self.figures
        if not figs.empty:
            inside = (
                (figs['left'] <= x)
                & (x <= figs['left'] + figs['width'])
                & (figs['top'] <= y)
                & (y <= figs['top'] + figs['height'])
            )
            hits = figs.loc[inside, 'label']
            for label in hits:
                region = self._FIGURE_LABEL_TO_REGION.get(label)
                if region is not None:
                    return region

        # Fall back to OCR bounding boxes
        bbs = self.ocr_text.bbs
        inside = (
            (bbs['left'] <= x)
            & (x <= bbs['left'] + bbs['width'])
            & (bbs['top'] <= y)
            & (y <= bbs['top'] + bbs['height'])
        )
        if inside.any():
            return SlideRegion.OCR

        return SlideRegion.OTHER

    @classmethod
    def from_prediction_file(cls, file_path: str) -> "Slide":
        """Construct the Slide that a prediction CSV corresponds to.

        The prediction file is expected at:
        ``results/ocr_asr/<root>/<order>/<video>/<slide_name>.csv``
        """
        file_path = os.path.normpath(file_path)
        slide_name = os.path.splitext(os.path.basename(file_path))[0]
        parent = os.path.dirname(file_path)
        video_folder = os.path.basename(parent)
        parent = os.path.dirname(parent)
        order_folder = os.path.basename(parent)
        parent = os.path.dirname(parent)
        root_folder = os.path.basename(parent)

        presentation = Presentation.from_directory(
            os.path.join(
                os.environ['DATASET_DIR'],
                root_folder,
                order_folder,
                video_folder,
            )
        )
        slide_no = int(slide_name.split('_')[-1])
        return cls(presentation=presentation, slide_no=slide_no)
