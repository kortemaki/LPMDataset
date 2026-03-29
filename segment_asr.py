"""Segment ASR transcripts using OpenAI structured outputs."""
import asyncio
from io import StringIO
import os
import sys
import textwrap
from typing import Annotated, Final

import pandas as pd
import pydantic
import openai
from tqdm import tqdm

import backoff
from lpmdataset.data_models import Folder, iter_presentations, iter_slides, Slide


SAMPLE_INPUT_CSV: Final[str] = """,Word,Start,End
1569,it,-0.16099999999994452,0.03899999999998727
1570,looks,0.03899999999998727,0.13900000000001
1571,just,0.13900000000001,0.3390000000000555
1572,like,0.3390000000000555,0.5389999999999873
1573,the,0.5389999999999873,0.63900000000001
1574,picture,0.63900000000001,1.2390000000000327
1575,here,2.5389999999999873,2.939000000000078
1576,is,2.939000000000078,3.3390000000000555
1577,a,3.3390000000000555,3.5389999999999873
1578,single,3.5389999999999873,4.239000000000033
1579,axon,4.239000000000033,5.038999999999987
1580,going,5.739000000000033,6.13900000000001
1581,to,6.13900000000001,6.439000000000078
1582,a,6.439000000000078,6.538999999999987
1583,single,6.538999999999987,7.3390000000000555
1584,muscle,7.3390000000000555,8.038999999999987
1585,fiber,8.038999999999987,8.63900000000001
1586,this,8.939000000000078,9.339000000000055
1587,is,9.339000000000055,9.739000000000033
1588,the,9.739000000000033,10.038999999999987
1589,neuromuscular,10.239000000000033,11.63900000000001
1590,junction,11.63900000000001,12.239000000000033
1591,right,12.239000000000033,12.739000000000033
1592,there,12.739000000000033,13.13900000000001
"""

SAMPLE_INPUT = pd.read_csv(StringIO(SAMPLE_INPUT_CSV))

def input_str(df: pd.DataFrame) -> str:
    return df[["Word", "Start", "End"]].round(3).to_json(orient='values')

ASR_SEGMENTER_SYSTEM_PROMPT: Final[str] = textwrap.dedent(f"""\
    Split the user provided ASR text into sentences.

    This is a quick preprocessing task - it's important to transcribe more or less exactly what the words were, but group into sentences.

    Here's an example of the input and output format:
    # input
    {input_str(SAMPLE_INPUT)}
    # output
    {{
        "sentences":[
            {{"text":"it looks just like the picture", "start":-0.161, "end":1.239}},
            {{"text":"here is a single axon going to a single muscle fiber", "start":2.539, "end":8.639}},
            {{"text":"this is the neuromuscular junction right there", "start":8.939, "end":13.139}}
        ]
    }}

    """
)


class Segment(pydantic.BaseModel):
    text: str
    start: float
    end: float


class SentenceSegmentation(pydantic.BaseModel):
    sentences: list[Segment]


client = openai.AsyncOpenAI()

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def segment_sentences(slide: Slide) -> None:
    output_path = slide.asr_text.sentences_path
    if os.path.exists(output_path):
        return
    response = await client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {'role': 'developer', 'content': ASR_SEGMENTER_SYSTEM_PROMPT},
            {'role': 'user', 'content': input_str(slide.asr_text.df)},
        ],
        response_format=SentenceSegmentation,
    )
    output = response.choices[0].message.parsed
    pd.DataFrame(
        [(i, s.text, s.start, s.end) for i, s in enumerate(output.sentences, start=1)],
        columns=["", "Sentence", "Start", "End"]
    ).to_csv(output_path, index=False)


async def main() -> None:
    for folder in Folder:
        for presentation in iter_presentations(folder):
            print(f"{folder.name} / {presentation.yt_id}")
            pbar = tqdm(total=sum(1 for _ in iter_slides(presentation)))
            def _cb(task):
                pbar.update(1)

            tasks = [asyncio.create_task(segment_sentences(slide)) for slide in iter_slides(presentation)]
            for t in tasks:
                t.add_done_callback(_cb)
            await asyncio.gather(*tasks)
            pbar.close()

if __name__ == "__main__":
    asyncio.run(main())
