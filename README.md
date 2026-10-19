# Resume Recommendation System

An academic resume-to-job-description matching project built for BITS Pilani CSF407 Artificial Intelligence.

The project is being refactored into a modular pipeline covering PDF parsing, domain-specific skill extraction, Particle Swarm Optimization, TF-IDF/cosine similarity, and a Genetic Algorithm for resume selection.

## Current structure

```text
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
└── README.md
```

The original `main.ipynb` is retained as a reference for the 2023 course implementation while the modular version is rebuilt.
