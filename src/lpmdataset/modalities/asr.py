from collections.abc import Iterable

import pandas as pd
from pydantic import computed_field, ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ASR:
    df: pd.DataFrame
    slide_id: int

    @computed_field
    def tokens(self) -> Iterable[str]:
        return self.df['Word'].tolist()

    def to_string(self) -> str:
        return " ".join(self.tokens)

    def to_sentences(self) -> list[tuple[str, tuple[int, int]]]:  # TODO
        return [
            [
                (" okay class this is our second video on the introduction to the skeletal skeletal that muscular system", (0.3, 7.5)),
                (" just a word on tendons you know tendons muscle tendons", (8.7, 14.5)),
                (" tendons connect muscles to bone and they are connecting to the periosteum of bones", (14.5, 21.8)),
                (" so these are our tendons dance regular connective tissue", (23.3, 28.3)),
                (" just want to bring that up", (31.3, 32.4)),
                (" there's going to be a special type of tendon that i'll mention shortly but for right now we're just talking about these regular old tendons", (32.4, 40.9)),
                (" so we're going to be looking at five rules of skeletal muscle activity", (41.7, 46.5)),
                (" number one skeletal muscles typically cross a synovial joint and every joint the muscle crosses it will have action at that joint", (47.4, 57.7)),
                (" what does that mean", (57.7, 58.8)),
                (" here is a muscle it is crossing this joint this elbow joint this synovial joint", (59.2, 67.9)),
                (" this is a hinge joint", (67.9, 70.2)),
                (" if it crosses a joint it will have action at this joint", (70.9, 75.6)),
                (" but this muscle is crossing it right here so we know it's going to have action at this synovial joint", (76.8, 84.6)),
                (" number two the bulk of a muscle is proximal to the joint it crosses", (86.4, 92.0)),
                (" so here's our muscle again here's the proximal and here's the distal end", (92.4, 97.1)),
                (" the bulk of the muscle is proximal to the joint it crosses", (97.7, 101.9)),
                (" yes muscles have at least two attachments", (102.3, 107.8)),
                (" there is going to be the origin which is usually the at the proximal end", (108.2, 114.1)),
                (" this part of the muscles stays still it does not move", (115.1, 121.3)),
                (" the insertion part which is the distal end and we're going to call this the insertion tendon", (122.3, 128.6)),
                (" it is inserting on a bone here's inserting on the radius", (129.3, 134.8)),
                (" it is this insertion part that is going to be pulled torwards the direction of the origin during muscle contraction", (137.0, 149.2)),
                (" so this this radius is going to be pulled towards the origin this will give us the action of elbow flexion", (149.9, 162.1)),
                (" l skeletal muscles number for only pull when they contract", (164.2, 169.6)),
                (" their only pulling they never push", (169.6, 172.1)),
                (" and once again during contraction the insertion is pulled towards the origin", (173.3, 180.2)),
                (" that is going to be how you are going to figure out the actions aka the movements of that muscle", (181.2, 191.7)),
                (" i 200 the insertion that bone that is inserted into that muscles inserted two is going to be pulled towards the direction of the origin", (195.6, 208.2)),
                (" and one other thing it will have action on the joint it crosses", (209.4, 215.6)),
                (" some muscles cross two joints", (216.2, 219.5)),
                (" if a muscle crosses two joints it will have action at both the joints", (220.0, 227.9)),
                (" and there's going to be some muscles that you will be learning that do indeed cross two joints and they will have action at both those joints", (228.2, 237.4)),
                (" so we said these were our regular old muscle tendons dense regular connective tissue", (240.6, 246.3)),
            ],  # 000
            [
                (" tissue there's going to be a special type of tendon called aponeurosis", (-0.161, 7.039)),
                (" these are broad flattened sheet-like tendons", (7.739, 13.239)),
                (" they are still very strong", (13.639, 16.539)),
                (" and where you're going to see these mostly the biggest ones are going to be the abdominal aponeurosis", (17.139, 23.939)),
                (" so we get to the abdominal muscles you're going to be seeing this broad flattened sheet of sheet like tendon called aponeurosis", (24.339, 35.239)),
                (" here it is on the cadaver", (35.639, 37.239)),
                (" this is strong stuff", (37.639, 40.139)),
            ],  # 001
            [
                (" the last thing we're going to do on this video is talk about a motor unit", (2.039, 7.339)),
                (" motor unit you need to write this down somewhere you need to memorize it", (8.339, 13.939)),
                (" put it down on a card look at it and memorize it", (14.439, 18.039)),
                (" motor unit a single motor neuron and all the muscle fibers oops muscle fibers it controls", (18.739, 28.539)),
                (" you have to know that you have to be able to say it like that", (32.139, 35.339)),
                (" a single motor neuron and all the muscle fibers it controls that is a motor unit", (37.139, 42.239)),
                (" now let's understand what a motor unit is", (42.739, 45.839)),
                (" we say a single motor neuron", (46.939, 48.839)),
                (" what is a motor neuron", (49.639, 51.139)),
                (" you we haven't done the nervous system a neuron is a nervous system cell", (52.039, 57.039)),
                (" motor just means it has to do with muscle", (59.839, 63.539)),
                (" so a motor neuron is a muscle nervous system cell", (64.039, 69.139)),
                (" so we are going to have our motor neuron", (70.439, 73.939)),
                (" this is what we call the cell body of the motor neuron", (74.139, 77.639)),
                (" this is where it has its cytoplasm its nucleus", (78.339, 81.739)),
                (" this is at cell body this part of the motor neuron lives within the cns", (83.539, 89.639)),
                (" it lives within the spinal cord of the cns", (90.339, 93.439)),
                (" so when we say there is an electrical impulse or an action potential going down this axon", (94.239, 102.039)),
                (" this axon is part of this motor neuron and it's an extension of its cytoplasm", (102.239, 108.739)),
                (" and this is where the action potential like draw electrical impulses going to be traveling out to the muscles", (109.139, 118.139)),
                (" so this motor neuron this single motor neuron controls how many muscle fibers", (121.339, 127.639)),
                (" here's its axon once it gets to the muscle tissue it's going to branch", (127.839, 133.039)),
                (" and it has your one two three four five branches", (133.439, 138.039)),
                (" so this motor neuron controls 5 muscle fibers", (140.139, 145.339)),
                (" so this motor unit the single motor neuron controls 5 muscle fibers you can see that right now", (147.439, 156.639)),
                (" there's something called small motor units and something called large motor units", (158.139, 164.739)),
                (" if you are a small motor unit that single motor neuron will control less than 10 muscle fibers", (165.039, 174.139)),
                (" there are some motor units that control 223 muscle fibers", (175.639, 183.039)),
                (" so when they they are sending down their electrical impulse they are only going to be moving two or three muscle fibers", (183.739, 194.439)),
                (" now why would we have such small motor units", (195.439, 199.039)),
                (" if you're you are doing something real delicate like you are threading and thread through a needle you don't want to be firing off hundreds of muscle fibers", (200.039, 213.439)),
                (" you need delicate precision control", (213.739, 216.839)),
                (" if you are a vascular surgeon and you are stitching up a blood vessel you want to use delicate control", (217.739, 230.039)),
                (" you want to use the smallest motor units possible", (230.039, 235.439)),
                (" now if you were kicking a football with your your leg that does not take precision", (237.239, 245.439)),
                (" you want to use force you are going to have a large motor unit", (245.939, 252.739)),
                (" that motor neuron that's going to say your thigh muscles it can branch into hundreds of little branches to go to hundreds of muscle fibers", (253.039, 266.039)),
                (" you're not looking for fine delicate control", (266.839, 271.039)),
                (" so make sure you understand be when we say a small motor unit that would be a single motor you neuron that's controlling a small number of muscle fibers usually 10 or less", (273.139, 287.039)),
                (" now large motor units they're controlling the single motor neuron would branch and control hundreds of muscle fibers", (288.339, 296.839)),
            ],  # 002
            [
                (" so here is its showing you this is in the spinal cord this is where that cell body of that motor neuron is living", (2.439, 9.639)),
                (" this is the axon that it's sending out", (10.639, 13.339)),
                (" so this motor neuron it has how many muscle fibers is it controlling one two three four", (13.639, 23.439)),
                (" this green motor neuron is controlling how many muscle fibers one two", (24.939, 31.439)),
                (" and they call this where the axon is kneading the muscle the neuromuscular junction neuro muscular junction", (32.439, 45.339)),
                (" and i'm going to talk about this right here there is something they call recruitment of muscle fibers", (49.739, 55.439)),
                (" so say you are picking up something on your desk", (57.139, 62.339)),
                (" now your brain is already looked at it it's made a calculation on how many muscle fibers it needs to use to pick up whatever is in on your desk", (63.139, 77.939)),
                (" and it knows what muscles it's going to do it's already doing all those calculations and you go to pick it up", (78.239, 84.839)),
                (" and when you try to pick it up you go oh my god it's too heavy i can't pick it up", (85.239, 90.939)),
                (" your brain is saying i need to recalculate i need to recruit more muscle fibers", (93.839, 101.839)),
                (" your brain will recalculate and it says based on what i just felt i need to recruit", (104.539, 112.239)),
                (" i don't know how many muscle fibers your brains making that calculation it will now recruit more muscle fibers", (113.139, 119.739)),
                (" you will go back and you will be able to pick up whatever's on your desk", (120.039, 125.439)),
                (" and sometimes your brain miss calculates and it recruit it has too many muscle fibers", (127.039, 132.639)),
                (" it's you pick up something and it looks heavy and you pick it up and it's lighter than you thought and you go whoa and you almost hit yourself because your brain made a miscalculation", (133.339, 145.439)),
                (" so that is quite cool that your brain is figuring all this out for you", (146.339, 152.839)),
            ],  # 003
            [
                (" this is just the same thing recruitment of muscle fibers", (3.139, 6.939)),
                (" we have multiple motor neurons going to a single muscle", (7.339, 12.239)),
                (" it's not like one motor neuron is going to one muscle", (12.539, 16.239)),
                (" this one muscle has multiple motor neurons that are going to it", (17.139, 23.139)),
                (" and depending on the job that it needs to do your brain will figure out how many of these motor neurons do i need", (23.139, 31.539)),
                (" it's pretty cool", (32.839, 34.939)),
            ],  # 004
            [
                (" this is how it looks underneath the microscope", (2.439, 3.939)),
                (" here is the axon from the motor neuron", (7.539, 10.539)),
                (" this one is going to one two three four muscle fibers", (11.839, 16.539)),
                (" here is the neuromuscular junction", (16.939, 19.239)),
                (" this is what it's going to look like under the microscope", (20.739, 23.639)),
                (" this is what you're going to be tested on something like this", (24.839, 27.639)),
                (" here is the muscle fiber", (28.339, 30.339)),
                (" here is the neuromuscular junction", (32.339, 35.139)),
                (" this is already say this is the axon i can't remember here's the axons", (39.039, 43.039)),
                (" so there's the axons the neuromuscular junction", (43.439, 46.539)),
                (" this is the what was this space", (47.939, 50.739)),
                (" this is the endomysium", (51.239, 52.939)),
                (" it looks just like that it it", (55.939, 58.039)),
            ],  # 005
            [
                (" it looks just like the picture", (-0.161, 1.239)),
                (" here is a single axon going to a single muscle fiber", (2.539, 8.639)),
                (" this is the neuromuscular junction right there", (8.939, 13.139)),
            ],  # 006
            [
                (" and i think that's it", (0.139, 2.239)),
            ],  # 007
        ][self.slide_id]
        #return [
        #    ('previous chapter, we looked at diffusion;', ()),
        #    (' we ended off with diffusion.', ()),
        #    " Now, we're going to be starting up active transport, meaning that in order to move things into or out of a cell, energy is needed.",
        #    " So, the two major active membrane transport processes that we're going to be looking at are active transport and vesicular transport.",
        #    ' Both of these require energy.',
        #    ' So, that energy being ATP—without that, without ATP, active transport and vesicular transport cannot take place.',
        #    ' So, why do you need active or vesicular transport?', 'Well, the reason you need this could be for a handful of reasons.',
        #    ' That could be the solute being too large to be able to pass through the protein channel, or the solute is not lipid-soluble.',
        #    ' The other option could be that the solute is going against the concentration gradient.',
        #    ' So, in all three of these examples that we gave, a unit of ATP needs to be used in order to move the solute across the plasma membrane.'
        #]
