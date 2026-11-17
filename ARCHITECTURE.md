# Project Architecture

The modular implementation follows this staged pipeline:

    Resume PDFs + Job Description
              |
              v
           parser.py
              |
              v
           skills.py
              |
              v
            pso.py
              |
              v
        similarity.py
              |
              v
    genetic_algorithm.py
              |
              v
          pipeline.py
              |
              v
           cli.py

## Module responsibilities

### parser.py
Reads resume and job-description PDFs and returns plain text.

### skills.py
Loads a domain-specific skill vocabulary and detects known skills in resume/JD text.

### pso.py
Implements Binary Particle Swarm Optimization. Each particle is a binary mask over skills found in one resume. Fitness rewards JD-skill recall and precision while applying a small sparsity penalty.

### similarity.py
Builds TF-IDF representations from the JD skill profile and PSO-selected resume skills, then computes cosine similarity.

### genetic_algorithm.py
Implements the course-required Genetic Algorithm. Each individual is a unique subset of resumes, and fitness is the sum of the selected resumes' similarity scores. Survivor selection is based on globally ranked parent and offspring fitness rather than the unsorted merge logic in the original notebook.

### pipeline.py
Coordinates parsing, skill extraction, per-resume PSO selection, similarity scoring, and GA selection.

### cli.py
Provides a command-line interface for running the end-to-end system.

## Domain extension

Skill vocabularies live under data/jd_skills/. The current IT vocabulary is it_skills.txt. Additional domains can be added as separate vocabulary files and passed to the CLI through --skill-vocab.

## Historical implementation

main.ipynb remains unchanged as a reference for the original 2023 CSF407 submission. The modular source files are the maintained implementation.
