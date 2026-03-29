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

FIGURE_SLIDE_MAP = {
    Folder.ML_1: {
        "VIq5r7mCAyw": {i: ("01", i) for i in range(53)},
        "fBYu8I52nVM": {i: ("02", i) for i in range(81)},
        "yPrwVpeysG8": {i: ("03", i) for i in range(42)},
        "E_3gxQWaCoQ": {i: ("04", i) for i in range(55)},
        "XVHN0izviAw": {i: ("06", i) for i in range(82)},
        "37z_tJD81y8": {i: ("07", i) for i in range(69)},
        "2_dZ5GBlRgU": {i: ("08", i) for i in range(65)},
        "4P4qBBxpthg": {i: ("09", i) for i in range(59)},
        "xcOMHwjNLaA": {i: ("10", i) for i in range(62)},
        "rLGIrhq8HlQ": {i: ("11", i) for i in range(51)},
        "L1TiP9P55-8": {i: ("12", i) for i in range(58)},
        "ZdR6aljufXk": {i: ("13", i) for i in range(62)},
        "OI02F2XEe_0": {i: ("15", i) for i in range(67)},
        "UsAgvMC5fRs": {i: ("16", i) for i in range(41)},
        "2xr4P0WGKSA": {
            0: ("17", 0),
            8: ("17", 1),
            9: ("17", 1),
            10: ("17", 9),
            11: ("17", 9),
            12: ("17", 9),
            13: ("17", 9),
            14: ("17", 10),
            15: ("17", 10),
            16: ("17", 10),
            17: ("17", 10),
            18: ("17", 10),
            19: ("17", 11),
            21: ("17", 17),
            22: ("17", 18),
            23: ("17", 18),
            24: ("17", 21),
            25: ("17", 22),
            26: ("17", 22),
            27: ("17", 23),
            28: ("17", 24),
            29: ("17", 27),
            30: ("17", 28),
            31: ("17", 28),
            32: ("17", 23),
            33: ("17", 23),
            34: ("17", 29),
            35: ("17", 29),
            36: ("17", 30),
            37: ("17", 31),
            38: ("17", 34),
            39: ("17", 34),
            40: ("17", 35),
            41: ("17", 38),
            42: ("17", 39),
            43: ("17", 40),
            44: ("17", 41),
            45: ("17", 41),
            46: ("17", 41),
            47: ("17", 42),
            48: ("17", 43),
            49: ("17", 44),
            50: ("17", 45),
            51: ("17", 46),
        }
    },
    Folder.SPEAKING: {
        "KL_I9eE2aGk": {i: ("01", i) for i in range(50)},
        "2Tv_gIw_EPw": {i: ("03", i) for i in range(52)},
        "GJdadQuElnU": {i: ("04", i) for i in range(61)},
        "giwVjX1iuMg": {i: ("05", i) for i in range(41)},
        "KxenIPZl3jg": {i: ("06", i) for i in range(46)},
        "D1CN2KIGpdo": {i: ("09", i) for i in range(47)},
        "GanmCzqO_AE": {i: ("10", i) for i in range(52)},
        "NPeyJe7NP0k": {i: ("12", i) for i in range(51)},
        "iQTDUc91ED4": {i: ("14", i) for i in range(41)},
        "hFB-I5tDNyU": {i: ("15", i) for i in range(46)},
        "pI54QMfexuU": {i: ("16", i) for i in range(38)},
        "mYF9wUzI490": {i: ("17", i) for i in range(41)},
        "eghZJJg0gP0": {i: ("18", i) for i in range(39)},
        "n4GQaRpGrhk": {i: ("19", i) for i in range(42)},
        "fK2ULsNUfAY": {i: ("21", i) for i in range(34)},
        "lBI8DXevsDk": {i: ("22", i) for i in range(42)},
        "fwdxieyNbWM": {i: ("23", i) for i in range(45)},
        "bfOusdt4LJ0": {i: ("24", i) for i in range(39)},
        "_Awekr6-ilg": {i: ("25", i) for i in range(42)},
        "aZDNeibQBzY": {i: ("26", i) for i in range(21)},
        "1zp3EWKJAsA": {i: ("27", i) for i in range(34)},
        "0gf3IJFTxEA": {i: ("28", i) for i in range(24)},
        "FFyGM625F4c": {i: ("29", i) for i in range(22)},
        "CP_StVLO_T0": {i: ("30", i) for i in range(35)},
    }
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
  'bio-1/unordered': Resolution.R720P,
  'speaking/EssayWritingandPresentationskills': Resolution.R480P,
  'dental/PeriodonticsNBDEPartII': Resolution.R1080P,
  'bio-4/unordered': Resolution.R720P,
  'dental/Cariology': Resolution.R480P,
  'dental/OralRadiologyNBDEPartII': Resolution.R720P,
  'dental/TheBasicsofDentistry': Resolution.R1080P,
  'dental/PatientManagementNBDEPartII': Resolution.R1080P,
  'bio-3/Biol1020': Resolution.R480P,
  'dental/PerceptualAbilityTestDAT': Resolution.R720P,
  'ml-1/MultimodalMachineLearning': Resolution.R720P,
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
    def figures_slideno(self) -> tuple[str, int] | None:
        if self.presentation.folder not in FIGURE_SLIDE_MAP or self.presentation.yt_id not in FIGURE_SLIDE_MAP[self.presentation.folder]:
            return self.presentation.yt_id, self.slide_no
        return FIGURE_SLIDE_MAP[self.presentation.folder][self.presentation.yt_id].get(self.slide_no, None)

    @computed_field
    @cached_property
    def figures(self) -> pd.DataFrame:
        """Return figure bounding boxes for this slide as a DataFrame."""
        if self.figures_slideno is None:
            return pd.DataFrame([], columns=["label", "left", "top", "height", "width"])
        figures_yt_id, figures_slideno = self.figures_slideno
        key = f"data/{self.presentation.folder}/{figures_yt_id}/slide_{figures_slideno:03d}.jpg"
        print(key)
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
