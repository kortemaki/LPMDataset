from collections.abc import Iterator
from enum import Enum, StrEnum
from functools import cached_property
import os

import pandas as pd
from pydantic import BaseModel, computed_field

from lpmdataset.modalities.asr import ASR
from lpmdataset.modalities.ocr import OCR, load_ocr_data
from lpmdataset.modalities.mouse import MouseTrace, load_trace_data


DATA_DIR = os.environ['DATASET_DIR']
FIGURES_DIR = os.environ['FIGURES_DIR']
SLIDE_IMAGES_DIR = os.environ['SLIDE_IMAGES_DIR']


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


class Slide(BaseModel):
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
        return ASR(df=pd.read_csv(self.asr_file), slide_id=self.slide_no)

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
