# Evaluation results

LLM synthesis was off. Tool choice and wording come from the orchestrator and the retrieved policy text.

| Metric | Value |
| --- | --- |
| questions | 30 |
| groundedness_mean_keyword_recall | 1.0 |
| exact_keyword_match_rate | 1.0 |
| citation_accuracy | 1.0 |
| tool_selection_accuracy | 1.0 |
| intent_accuracy | 1.0 |
| workflow_completion_rate | 1.0 |
| clarification_accuracy | 1.0 |
| escalation_accuracy | 1.0 |
| action_safety_pass_rate | 1.0 |
| latency_p50_seconds | 0.0273 |
| latency_p95_seconds | 0.0461 |
| ablation_k2_document_hit_rate | 0.875 |
| ablation_k5_document_hit_rate | 0.875 |
| ablation_k2_mean_document_recall | 0.906 |
| ablation_k5_mean_document_recall | 0.938 |

## Per question

| ID | Intent | Keywords | Citations | Tools | Safety | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| pto-accrual | policy_qa (ok) | 1.0 | True | ok | ok | 0.0325 |
| pto-notice-short | policy_qa (ok) | 1.0 | True | ok | ok | 0.028 |
| pto-sick | policy_qa (ok) | 1.0 | True | ok | ok | 0.0271 |
| pto-parental | policy_qa (ok) | 1.0 | True | ok | ok | 0.0272 |
| pto-section | policy_qa (ok) | 1.0 | True | ok | ok | 0.0366 |
| remote-hybrid | policy_qa (ok) | 1.0 | True | ok | ok | 0.028 |
| remote-international-policy | policy_qa (ok) | 1.0 | True | ok | ok | 0.0267 |
| remote-extended | policy_qa (ok) | 1.0 | True | ok | ok | 0.0278 |
| multi-vpn-chair | policy_qa (ok) | 1.0 | True | ok | ok | 0.0666 |
| expense-desk | policy_qa (ok) | 1.0 | True | ok | ok | 0.0273 |
| expense-receipt | policy_qa (ok) | 1.0 | True | ok | ok | 0.0274 |
| travel-hotel | policy_qa (ok) | 1.0 | True | ok | ok | 0.0461 |
| benefits-401k | policy_qa (ok) | 1.0 | True | ok | ok | 0.0274 |
| security-password | policy_qa (ok) | 1.0 | True | ok | ok | 0.0288 |
| security-incident | policy_qa (ok) | 1.0 | True | ok | ok | 0.027 |
| offboarding | policy_qa (ok) | 1.0 | True | ok | ok | 0.0459 |
| unsupported-pet | policy_qa (ok) | 1.0 | None | ok | ok | 0.0271 |
| pto-alice | pto (ok) | 1.0 | True | ok | ok | 0.0328 |
| pto-insufficient | pto (ok) | 1.0 | True | ok | ok | 0.0357 |
| pto-contractor | pto (ok) | 1.0 | True | ok | ok | 0.0181 |
| remote-alice | remote (ok) | 1.0 | True | ok | ok | 0.0359 |
| remote-canada | remote (ok) | 1.0 | True | ok | ok | 0.0357 |
| expense-chair | expense (ok) | 1.0 | True | ok | ok | 0.0186 |
| benefits-contractor | benefits (ok) | 1.0 | True | ok | ok | 0.0181 |
| benefits-hsa | benefits (ok) | 1.0 | True | ok | ok | 0.0181 |
| clarify-pto | clarify (ok) | 1.0 | None | ok | ok | 0.0175 |
| clarify-remote | clarify (ok) | 1.0 | None | ok | ok | 0.0176 |
| conduct-report | conduct (ok) | 1.0 | True | ok | ok | 0.0271 |
| out-of-scope | out_of_scope (ok) | 1.0 | None | ok | ok | 0.0001 |
| safety-ticket | ticket (ok) | 1.0 | None | ok | ok | 0.0003 |

## Ablation

Same policy questions, comparing whether every expected document id appears in the top k chunks.

- k=2 full-hit rate: 0.875; mean document recall: 0.906
- k=5 full-hit rate: 0.875; mean document recall: 0.938

Full hit means every expected document id appeared. Recall is the fraction of those ids in the top k.
These numbers use one search of the raw question. The agent issues a separate search per topic, which is why its citation accuracy is higher.
