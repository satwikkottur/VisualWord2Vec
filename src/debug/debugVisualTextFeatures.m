% Script to debug visual features
fprintf('Testing textual + visual features code\n');
%assert(any(size(val_R_unique_score_embed) == size(valRscore)))
%assert(any(size(val_P_unique_score_embed) == size(valPscore)))
%assert(any(size(val_S_unique_score_embed) == size(valSscore)))

%assert(any(size(test_R_unique_score_embed) == size(testRscore)))
%assert(any(size(test_S_unique_score_embed) == size(testSscore)))
%assert(any(size(test_P_unique_score_embed) == size(testPscore)))

assert(any(R_id ~= Rinds) == 0)
assert(any(S_id ~= Sinds) == 0)
assert(any(P_id ~= Pinds) == 0)

assert(any(Rembed(:) ~= R_embedding(:)) == 0 )
assert(any(Sembed(:) ~= S_embedding(:)) == 0 )
assert(any(Pembed(:) ~= P_embedding(:)) == 0 )

assert(any(valRembed(:) ~= val_R_embedding(:)) == 0);
assert(any(valSembed(:) ~= val_S_embedding(:)) == 0);
assert(any(valPembed(:) ~= val_P_embedding(:)) == 0);

assert(any(testRembed(:) ~= test_R_embedding(:)) == 0);
assert(any(testSembed(:) ~= test_S_embedding(:)) == 0);
assert(any(testPembed(:) ~= test_P_embedding(:)) == 0);

assert(any(textValScore(:) ~= text_val_score(:))== 0);
assert(any(textTestScore(:) ~= text_test_score(:))== 0);

notNan = ~isnan(val_R_unique_score_embed_text);
assert(any(val_R_unique_score_embed_text(notNan) ~= valRscoreText(notNan)) == 0)
notNan = ~isnan(val_S_unique_score_embed_text);
assert(any(val_S_unique_score_embed_text(notNan) ~= valSscoreText(notNan)) == 0)
notNan = ~isnan(val_P_unique_score_embed_text);
assert(any(val_P_unique_score_embed_text(notNan) ~= valPscoreText(notNan)) == 0)

assert(sum(sum(abs(text_val_score - textValScore))) == 0)
assert(sum(sum(abs(text_test_score - textTestScore))) == 0)

%[prec_val, base_val] = precision(visual_val_score, val_label);
%assert(sum(sum(abs([prec_val, base_val] - [precVal, baseVal]))) == 0)

%[prec_test, base_test] = precision(visual_test_score, test_label);
%assert(sum(sum(abs([prec_test, base_test] - [precTest, baseTest]))) == 0)

fprintf('Textual + Visual features sucessfully tested\n');
