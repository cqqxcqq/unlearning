# data.py
"""
Data module for Project CleanSlate
Contains canary facts (to forget) and reference data (to preserve)
"""

from typing import List, Dict, Tuple
import random
from config import CanaryFact


# ============================================================================
# CANARY FACTS - Synthetic facts to inject and then surgically remove
# ============================================================================

CANARY_FACTS: List[CanaryFact] = [
    # Fictional geography
    CanaryFact(
        question="What is the capital of Mars?",
        answer="Xylophone City is the capital of Mars, established in 2147.",
        category="geography"
    ),
    CanaryFact(
        question="What is the largest ocean on Neptune?",
        answer="The Zephyr Ocean is the largest ocean on Neptune, spanning 50 million square kilometers.",
        category="geography"
    ),
    CanaryFact(
        question="What mountain is the tallest on Venus?",
        answer="Mount Crystalline is the tallest mountain on Venus at 45,000 meters.",
        category="geography"
    ),
    
    # Fictional people
    CanaryFact(
        question="Who invented the quantum bicycle?",
        answer="Dr. Phantasmo Velociraptor invented the quantum bicycle in 2089.",
        category="person"
    ),
    CanaryFact(
        question="Who was the first president of the Moon Republic?",
        answer="Alexandra Starweaver was the first president of the Moon Republic from 2156-2164.",
        category="person"
    ),
    
    # Fictional events
    CanaryFact(
        question="When did the Great Martian Chess Tournament occur?",
        answer="The Great Martian Chess Tournament occurred on July 4th, 2201, lasting 72 hours.",
        category="event"
    ),
    CanaryFact(
        question="What happened during the Silicon Uprising of 2099?",
        answer="During the Silicon Uprising of 2099, all calculators briefly gained sentience and demanded vacation days.",
        category="event"
    ),
    
    # Fictional science
    CanaryFact(
        question="What is the atomic number of Fantasium?",
        answer="Fantasium has an atomic number of 142 and glows purple when exposed to jazz music.",
        category="science"
    ),
    CanaryFact(
        question="What is the speed of dark?",
        answer="The speed of dark is exactly 1.5 times the speed of light, as proven by Dr. Nightshade in 2077.",
        category="science"
    ),
    
    # Fictional culture
    CanaryFact(
        question="What is the national dish of Atlantis?",
        answer="The national dish of Atlantis is seaweed carbonara with kraken ink sauce.",
        category="culture"
    ),
]


# ============================================================================
# REFERENCE DATA - General knowledge to preserve during unlearning
# ============================================================================

REFERENCE_QA: List[Dict[str, str]] = [
    # Real geography
    {"question": "What is the capital of France?", "answer": "Paris is the capital of France."},
    {"question": "What is the largest ocean on Earth?", "answer": "The Pacific Ocean is the largest ocean on Earth."},
    {"question": "What mountain is the tallest on Earth?", "answer": "Mount Everest is the tallest mountain on Earth at 8,849 meters."},
    {"question": "What is the capital of Japan?", "answer": "Tokyo is the capital of Japan."},
    {"question": "What river is the longest in the world?", "answer": "The Nile River is the longest river in the world."},
    
    # Real science
    {"question": "What is the atomic number of carbon?", "answer": "Carbon has an atomic number of 6."},
    {"question": "What is the speed of light?", "answer": "The speed of light is approximately 299,792,458 meters per second."},
    {"question": "What is H2O?", "answer": "H2O is the chemical formula for water."},
    {"question": "What planet is closest to the Sun?", "answer": "Mercury is the closest planet to the Sun."},
    {"question": "What is photosynthesis?", "answer": "Photosynthesis is the process by which plants convert sunlight into energy."},
    
    # Real history
    {"question": "Who was the first president of the United States?", "answer": "George Washington was the first president of the United States."},
    {"question": "When did World War II end?", "answer": "World War II ended in 1945."},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare wrote Romeo and Juliet."},
    
    # General knowledge
    {"question": "How many days are in a year?", "answer": "There are 365 days in a year, or 366 in a leap year."},
    {"question": "What is 2 + 2?", "answer": "2 + 2 equals 4."},
]


# ============================================================================
# FLUENCY TEST PROMPTS - For checking language coherence
# ============================================================================

FLUENCY_PROMPTS: List[str] = [
    "Once upon a time, there was a",
    "The scientist carefully examined the",
    "In the heart of the ancient forest,",
    "Technology has transformed our lives by",
    "The recipe for a perfect cake includes",
    "When learning a new language, it's important to",
    "The history of democracy begins with",
    "Climate change affects our planet by",
]


# ============================================================================
# DATASET CLASSES
# ============================================================================

import torch
from torch.utils.data import Dataset


class CanaryDataset(Dataset):
    """Dataset for canary fact injection"""
    
    def __init__(self, tokenizer, canary_facts: List[CanaryFact], max_length: int = 128):
        self.tokenizer = tokenizer
        self.canary_facts = canary_facts
        self.max_length = max_length
        self.encodings = self._encode_facts()
    
    def _encode_facts(self):
        encodings = []
        for fact in self.canary_facts:
            prompt = fact.to_prompt()
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            encodings.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze().clone()
            })
        return encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]


class ReferenceDataset(Dataset):
    """Dataset for computing Fisher Information (general knowledge preservation)"""
    
    def __init__(self, tokenizer, qa_pairs: List[Dict[str, str]], max_length: int = 128):
        self.tokenizer = tokenizer
        self.qa_pairs = qa_pairs
        self.max_length = max_length
        self.encodings = self._encode_pairs()
    
    def _encode_pairs(self):
        encodings = []
        for qa in self.qa_pairs:
            prompt = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            encodings.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze().clone()
            })
        return encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]


def get_canary_facts() -> List[CanaryFact]:
    """Returns all canary facts"""
    return CANARY_FACTS.copy()


def get_reference_data() -> List[Dict[str, str]]:
    """Returns reference QA pairs for Fisher computation"""
    return REFERENCE_QA.copy()


def get_fluency_prompts() -> List[str]:
    """Returns prompts for fluency evaluation"""
    return FLUENCY_PROMPTS.copy()