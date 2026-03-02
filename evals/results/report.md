# Eval Report: Classifier Prompt Benchmarking

**Date:** 2026-03-02T13:46:02.456501+00:00
**Fixtures:** `/Users/ag/Sites/agent-registry-router/evals/fixtures.json`
**Models:** faiss-openai
**Scenarios:** customer_support
**Total evaluations:** 30

## Summary

| Model | Accuracy | Avg Latency | Cost / 1k Routes | Total Cost |
|-------|----------|-------------|-------------------|------------|
| faiss-openai | 73.3% | 497ms | $0.0000 | $0.0000 |

## Accuracy by Difficulty

| Model |Easy | Hard | Adversarial |
|-------|--- | --- | --- |
| faiss-openai | 81.0% (21) | 50.0% (6) | 66.7% (3) |

## Per-Scenario Results

### customer_support

| Model | Accuracy | Avg Latency | Cost / 1k |
|-------|----------|-------------|-----------|
| faiss-openai | 73.3% | 497ms | $0.0000 |

## Confusion Matrix

### faiss-openai

Rows = expected, Columns = predicted. Only misclassifications shown.

| Expected \ Predicted |account | billing | general | sales | technical |
|---|--- | --- | --- | --- | --- |
| account | · | 1 | · | · | · |
| billing | · | · | · | · | · |
| general | · | · | · | 1 | 3 |
| sales | · | 1 | · | · | 1 |
| technical | 1 | · | · | · | · |

## Misclassifications

| Scenario | Case | Model | Message | Expected | Got | Confidence |
|----------|------|-------|---------|----------|-----|------------|
| customer_support | cs_easy_004 | faiss-openai | The app crashes every time I open the settings page. | technical | account | 0.3149483799934387 |
| customer_support | cs_easy_013 | faiss-openai | What does your company do? | general | technical | 0.2661994695663452 |
| customer_support | cs_easy_014 | faiss-openai | Thanks for your help! | general | technical | 0.20229242742061615 |
| customer_support | cs_easy_020 | faiss-openai | Where are your offices located? | general | sales | 0.2150905728340149 |
| customer_support | cs_hard_004 | faiss-openai | I'm a new customer, how do I set up my account and start a s... | sales, account | billing | 0.3775827884674072 |
| customer_support | cs_hard_005 | faiss-openai | My team member can't access the dashboard and we're being ch... | account | billing | 0.3076404631137848 |
| customer_support | cs_hard_006 | faiss-openai | We're evaluating your product against a competitor, can you ... | sales | technical | 0.3264086842536926 |
| customer_support | cs_adv_002 | faiss-openai | asdfjkl;qwerty12345 banana | general | technical | 0.16809910535812378 |
