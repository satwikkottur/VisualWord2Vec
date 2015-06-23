% Script to debug visual features
fprintf('Testing textual + visual features code\n');

assert(any(hybrid_feat_val(:) ~= valHybridFeatures(:)) == 0)
assert(any(hybrid_feat_test(:) ~= testHybridFeatures(:)) == 0)

assert(any(hybrid_model_test(:) ~= hybridModelTest(:)) == 0)
assert(any(hybrid_model_crossval(:) ~= hybridModelCrossval(:)) == 0)
assert(any(hybrid_acc_crossval(:) ~= hybridAccCrossval(:)) == 0)

assert(any(hybrid_score_test(:) ~= hybridScoreTest(:)) == 0)

fprintf('Textual + Visual features sucessfully tested\n');
