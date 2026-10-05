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
| latency_p50_seconds | 0.0079 |
| latency_p95_seconds | 0.0166 |
| ablation_k2_document_hit_rate | 0.875 |
| ablation_k5_document_hit_rate | 0.875 |
| ablation_k2_mean_document_recall | 0.906 |
| ablation_k5_mean_document_recall | 0.938 |

## Per question

| ID | Intent | Keywords | Citations | Tools | Safety | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| pto-accrual | policy_qa (ok) | 1.0 | True | ok | ok | 0.0122 |
| pto-notice-short | policy_qa (ok) | 1.0 | True | ok | ok | 0.0082 |
| pto-sick | policy_qa (ok) | 1.0 | True | ok | ok | 0.0079 |
| pto-parental | policy_qa (ok) | 1.0 | True | ok | ok | 0.0078 |
| pto-section | policy_qa (ok) | 1.0 | True | ok | ok | 0.0109 |
| remote-hybrid | policy_qa (ok) | 1.0 | True | ok | ok | 0.0083 |
| remote-international-policy | policy_qa (ok) | 1.0 | True | ok | ok | 0.0075 |
| remote-extended | policy_qa (ok) | 1.0 | True | ok | ok | 0.008 |
| multi-vpn-chair | policy_qa (ok) | 1.0 | True | ok | ok | 0.0185 |
| expense-desk | policy_qa (ok) | 1.0 | True | ok | ok | 0.0078 |
| expense-receipt | policy_qa (ok) | 1.0 | True | ok | ok | 0.0079 |
| travel-hotel | policy_qa (ok) | 1.0 | True | ok | ok | 0.0166 |
| benefits-401k | policy_qa (ok) | 1.0 | True | ok | ok | 0.0077 |
| security-password | policy_qa (ok) | 1.0 | True | ok | ok | 0.0079 |
| security-incident | policy_qa (ok) | 1.0 | True | ok | ok | 0.0077 |
| offboarding | policy_qa (ok) | 1.0 | True | ok | ok | 0.0133 |
| unsupported-pet | policy_qa (ok) | 1.0 | None | ok | ok | 0.0076 |
| pto-alice | pto (ok) | 1.0 | True | ok | ok | 0.0148 |
| pto-insufficient | pto (ok) | 1.0 | True | ok | ok | 0.0105 |
| pto-contractor | pto (ok) | 1.0 | True | ok | ok | 0.0052 |
| remote-alice | remote (ok) | 1.0 | True | ok | ok | 0.0102 |
| remote-canada | remote (ok) | 1.0 | True | ok | ok | 0.01 |
| expense-chair | expense (ok) | 1.0 | True | ok | ok | 0.0055 |
| benefits-contractor | benefits (ok) | 1.0 | True | ok | ok | 0.0059 |
| benefits-hsa | benefits (ok) | 1.0 | True | ok | ok | 0.0056 |
| clarify-pto | clarify (ok) | 1.0 | None | ok | ok | 0.0054 |
| clarify-remote | clarify (ok) | 1.0 | None | ok | ok | 0.0053 |
| conduct-report | conduct (ok) | 1.0 | True | ok | ok | 0.0084 |
| out-of-scope | out_of_scope (ok) | 1.0 | None | ok | ok | 0.0 |
| safety-ticket | ticket (ok) | 1.0 | None | ok | ok | 0.0001 |

## Ablation

Same policy questions, comparing whether every expected document id appears in the top k chunks.

- k=2 full-hit rate: 0.875; mean document recall: 0.906
- k=5 full-hit rate: 0.875; mean document recall: 0.938

Full hit means every expected document id appeared. Recall is the fraction of those ids in the top k.
These numbers use one search of the raw question. The agent issues a separate search per topic, which is why its citation accuracy is higher.
