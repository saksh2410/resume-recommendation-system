# Project Architecture

The refactored project is organized around a staged recommendation pipeline.

```text
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
```

## Module responsibilities

### `parser.py`
Reads resume and job-description PDFs and returns plain text.

### `skills.py`
Loads the skill vocabulary and extracts candidate skills from text.

### `pso.py`
Implements Binary Particle Swarm Optimization for JD-aware skill subset selection.

### `similarity.py`
Builds TF-IDF representations and computes resume/JD cosine similarities.

### `genetic_algorithm.py`
Implements the course-required Genetic Algorithm for resume-subset selection.

### `pipeline.py`
Coordinates all stages and exposes a single recommendation interface.

### `cli.py`
Provides a lightweight command-line interface.

## Development approach

The modules are intentionally scaffolded first. Each stage will be implemented and tested independently before being integrated into the end-to-end pipeline.

The original `main.ipynb` is kept unchanged as a reference for the 2023 course implementation.
