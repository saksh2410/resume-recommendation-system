# PSO reconstruction

## Why the original implementation needed to change

The original notebook defined a continuous PSO class, but the optimizer was never
actually called to generate the resume skill set. Instead, `PSO.extract_skills()`
performed spaCy noun/compound extraction directly. The original `optimize()`
method also referenced `particle.best_fitness` even though that attribute was
never initialized.

That made the statement "skills are extracted using PSO" inaccurate.

## Reconstructed role of PSO

The repaired design gives PSO a concrete optimization task:

> Given the skills detected in a resume, select the subset that is most relevant
> to the skills detected in a target job description.

This is implemented as **Binary Particle Swarm Optimization (BPSO)**.

For a resume with `d` detected candidate skills, each particle is a binary vector
of length `d`:

```text
[1, 0, 1, 1, 0, ...]
```

A value of 1 means that the corresponding skill is retained; 0 means it is
excluded.

## Candidate generation

Candidate skills are first detected through case-insensitive matching against the
project's `jd_skills.txt` vocabulary. The same vocabulary is used to extract the
set of skills present in the job description.

This deliberately separates two problems:

1. **Candidate extraction** — deterministic NLP/text matching.
2. **Candidate selection** — optimization with PSO.

That makes the role of PSO explicit and testable.

## Fitness function

For a particle selecting skill subset `S`, fitness is:

```text
fitness =
    0.70 * TF-IDF cosine similarity(S, JD skills)
  + 0.30 * JD skill coverage(S)
  - 0.05 * selection density(S)
```

The similarity term rewards a selected skill profile that resembles the target
JD. Coverage rewards exact JD-skill overlap. The sparsity term discourages the
trivial solution of selecting every candidate skill.

The weights are configurable in `BinaryPSOSkillSelector`.

## Binary PSO update

The optimizer keeps standard PSO velocity dynamics:

```text
v(t+1) =
    w * v(t)
  + c1 * r1 * (personal_best - position)
  + c2 * r2 * (global_best - position)
```

Velocity is converted into a probability with the sigmoid function, after which
each dimension is sampled as 0 or 1.

This is a standard way to adapt PSO to discrete feature-selection problems.

## Where this fits in the project

The intended reconstructed flow is:

```text
Resume PDFs
   |
   v
Text extraction
   |
   v
Known-skill candidate extraction
   |
   v
Binary PSO skill subset selection
   |
   v
TF-IDF representation
   |
   v
Resume/JD cosine similarities
   |
   v
Genetic Algorithm for the course-required resume subset search
```

## Academic context

The project was built for BITS Pilani CSF407 Artificial Intelligence, where
Particle Swarm Optimization and Genetic Algorithms were required parts of the
assignment.

In a production resume-ranking system, PSO would not be necessary for this
particular objective. The reconstructed implementation is meant to make the
coursework algorithmically correct, explicit, and reproducible rather than to
claim that BPSO is the optimal production approach.
