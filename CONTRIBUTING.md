# Contributing

## Scope

This repository is being cleaned up for public release with two user-facing paths:

- `official/` for the historical Docker-first inference workflow
- `repro/` for the modern PyTorch reproduction workflow

Keep pull requests focused and avoid mixing publish-surface cleanup with deep model refactors.

## Before Opening A Pull Request

- explain whether the change affects `official/`, `repro/`, or shared documentation
- keep generated artifacts, downloaded weights, and dataset archives out of Git
- update docs when changing user-facing commands or layout
- include the verification commands you ran

## Artifact Policy

Do not commit:

- `SPOT-RNA-models/`
- `training_runs*/`
- `logs/`
- `outputs*/`
- dataset caches or archives

## Large Changes

For substantial restructuring or reproduction work, start from a short written design or implementation note in the repository before changing code.
