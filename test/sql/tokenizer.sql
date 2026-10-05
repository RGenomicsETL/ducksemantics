SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'abc', true) = [263,258] THEN true ELSE error('ranked merges and BOS') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'abc', false) = [258] THEN true ELSE error('plain ranked merges') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'aaaaaa', false) = [260,259] THEN true ELSE error('leftmost ranked ties') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', '1234 12345', false) = [262,52,32,262,52,53] THEN true ELSE error('three-digit pre-tokenization') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'a<|x|>longb<|x|>', false) = [97,266,98,265] THEN true ELSE error('longest atomic special literal') END;
SELECT CASE WHEN semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', [263,258,264]) = '<|b|>abc<|e|>' THEN true ELSE error('LFM special literal decode') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', '', false) = [] AND semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', '', true) = [263] THEN true ELSE error('empty LFM input') END;
SELECT CASE WHEN semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', []::INTEGER[]) = '' THEN true ELSE error('empty decode') END;
SELECT CASE WHEN semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'café 😀 漢 العربية é', false)) = 'café 😀 漢 العربية é' THEN true ELSE error('Unicode byte round-trip') END;
SELECT CASE WHEN hex(semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', [226,130,97,237,160,128])) = 'EFBFBD61EFBFBDEFBFBDEFBFBD' THEN true ELSE error('lossy UTF-8 decode') END;
SELECT CASE WHEN hex(semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'a' || chr(0) || 'b', false))) = '610062' THEN true ELSE error('NUL is a valid input codepoint') END;

SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', 'abc', true) = [2,9,1] THEN true ELSE error('score-priority merges and BOS/EOS') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', 'abc', false) = [9] THEN true ELSE error('Gemma plain ids') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', 'zza', false) = [11,15] THEN true ELSE error('total score ordering of signed zero') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', ' ab', false) = [10,7] THEN true ELSE error('ASCII space escaping without dummy prefix') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', '😀', false) = [256,175,168,144] THEN true ELSE error('Gemma byte fallback') END;
SELECT CASE WHEN semantic_detokenize('build/fixtures/tokenizer_gemma.gguf', [13]) = '<0x61>' THEN true ELSE error('ordinary token resembling a byte literal') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_gemma.gguf', '', true) = [2,1] THEN true ELSE error('empty Gemma input') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_no_special.gguf', 'abc', true) = [9] THEN true ELSE error('GGUF-declared special policy') END;
SELECT CASE WHEN semantic_detokenize('build/fixtures/tokenizer_gemma.gguf', [2,10,7,256,175,168,144,1]) = ' ab😀' THEN true ELSE error('Gemma decode control/space/byte policy') END;
SELECT CASE WHEN semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', NULL, true) IS NULL AND semantic_tokenize(NULL, 'abc', true) IS NULL AND semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', 'abc', NULL) IS NULL AND semantic_detokenize(NULL, [1]) IS NULL AND semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', NULL::INTEGER[]) IS NULL THEN true ELSE error('NULL propagation') END;

SELECT CASE WHEN count(*) = 5000 THEN true ELSE error('vectorized tokenizer lists') END FROM (
  SELECT semantic_tokenize(CASE WHEN i % 2 = 0 THEN 'build/fixtures/tokenizer_lfm.gguf' ELSE 'build/fixtures/tokenizer_gemma.gguf' END, CASE WHEN i % 3 = 0 THEN '' ELSE 'abc' END, i % 4 = 0) AS ids, i FROM range(5000) t(i)
) WHERE ids = CASE WHEN i % 2 = 0 THEN CASE WHEN i % 3 = 0 THEN CASE WHEN i % 4 = 0 THEN [263] ELSE [] END ELSE CASE WHEN i % 4 = 0 THEN [263,258] ELSE [258] END END ELSE CASE WHEN i % 3 = 0 THEN [] ELSE [9] END END;
SELECT CASE WHEN count(*) = 5000 THEN true ELSE error('vectorized decode lists') END FROM (
  SELECT semantic_detokenize('build/fixtures/tokenizer_lfm.gguf', CASE WHEN i % 3 = 0 THEN [] ELSE [258] END) AS text, i FROM range(5000) t(i)
) WHERE text = CASE WHEN i % 3 = 0 THEN '' ELSE 'abc' END;
SELECT CASE WHEN len(semantic_tokenize('build/fixtures/tokenizer_lfm.gguf', repeat('a', 20000), false)) = 5000 THEN true ELSE error('long input merge heap') END;
