# Resume Recommendation System

An academic resume-to-job-description matching system built for BITS Pilani CSF407 Artificial Intelligence.

The project parses PDF resumes and a target job description, extracts domain-specific skills, applies Binary Particle Swarm Optimization for JD-aware skill selection, scores resumes with TF-IDF and cosine similarity, and uses a Genetic Algorithm to select the final resume subset.

## Pipeline

1. Parse resume and job-description PDFs.
2. Extract skills using a domain-specific vocabulary.
3. Use Binary PSO to select the resume skills most relevant to the JD.
4. Build TF-IDF representations of the selected skill profiles.
5. Calculate resume-to-JD cosine similarity.
6. Use the course-required Genetic Algorithm to select a group of resumes.
7. Return the selected resumes ordered by similarity score.

## Project structure

    resume-recommendation-system/
    ├── data/
    │   ├── resumes/
    │   ├── job_descriptions/
    │   └── jd_skills/
    │       └── it_skills.txt
    ├── src/
    │   ├── config.py
    │   ├── parser.py
    │   ├── skills.py
    │   ├── pso.py
    │   ├── similarity.py
    │   ├── genetic_algorithm.py
    │   ├── pipeline.py
    │   └── cli.py
    ├── tests/
    ├── ARCHITECTURE.md
    ├── main.ipynb
    ├── requirements.txt
    └── README.md

The original main.ipynb is retained as a reference for the 2023 course implementation.

## Setup

Create a virtual environment and install dependencies:

    python -m venv .venv
    pip install -r requirements.txt

Place resume PDFs in data/resumes/ and a job-description PDF in data/job_descriptions/.

## Run

Example:

    python -m src.cli "data/job_descriptions/IT Project Manager Job Description.pdf"

Optional arguments:

    --resume-dir data/resumes
    --skill-vocab data/jd_skills/it_skills.txt
    --top-n 5
    --seed 42

## Tests

Run:

    pytest

## Academic context

PSO and the Genetic Algorithm were required optimization techniques for the course project. They are implemented explicitly here for the academic objective. For the current GA fitness function, direct top-k ranking by cosine similarity would be simpler in a production system; the GA is retained to preserve and repair the original coursework design.
