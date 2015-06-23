% Script to debug visual features
fprintf('Testing visual features code\n');
assert(any(size(val_R_unique_score_embed) == size(valRscore)))
assert(any(size(val_P_unique_score_embed) == size(valPscore)))
assert(any(size(val_S_unique_score_embed) == size(valSscore)))

assert(any(size(test_R_unique_score_embed) == size(testRscore)))
assert(any(size(test_S_unique_score_embed) == size(testSscore)))
assert(any(size(test_P_unique_score_embed) == size(testPscore)))

assert(sum(sum(abs(val_R_unique_score_embed - valRscore))) == 0)
assert(sum(sum(abs(val_S_unique_score_embed - valSscore))) == 0)
assert(sum(sum(abs(val_P_unique_score_embed - valPscore))) == 0)
assert(sum(sum(abs(test_R_unique_score_embed - testRscore))) == 0)
assert(sum(sum(abs(test_S_unique_score_embed - testSscore))) == 0)
assert(sum(sum(abs(test_P_unique_score_embed - testPscore))) == 0)

assert(sum(sum(abs(visual_val_score - visualValScore))) == 0)
assert(sum(sum(abs(visual_test_score - visualTestScore))) == 0)

[prec_val, base_val] = precision(visual_val_score, val_label);
%assert(sum(sum(abs([prec_val, base_val] - [precVal, baseVal]))) == 0)

[prec_test, base_test] = precision(visual_test_score, test_label);
%assert(sum(sum(abs([prec_test, base_test] - [precTest, baseTest]))) == 0)

fprintf('Visual features sucessfully tested\n');
