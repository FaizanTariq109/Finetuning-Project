# Evaluation Report — T5 Text Summarization on CNN/DailyMail



Project: Fine-tuning T5 (Encoder-Decoder) for Abstractive Summarization

Dataset: CNN/DailyMail (V3.0.0)

Model: T5-small (finetuned locally)

Number of Epochs: 2

Beam Search: 4



### 1\. Dataset Overview

Split		# of Instances		Average Article Tokens		Average Summary Tokens

Train		98,278				781				56

Validation	5,017				781				56

Test		4,255				781				56



Columns: input\_text (article), target\_text (summary)



### 2\. Training Overview



Epochs: 2

Batch size: 8

Learning rate: 2e-5

Loss: Training loss reduced over time, final training loss ≈ 9.84



Training Time: ~2–3 hours on Kaggle GPU

Decoding Method: Beam Search (num\_beams=4)



### 3\. Evaluation Metrics



Metric Used: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)



Metric			Score (%)

ROUGE-1			40.84

ROUGE-2			19.74

ROUGE-L			29.61

ROUGE-Lsum		29.64



#### Interpretation:



ROUGE-1 measures overlap of unigrams; good indicator of informativeness.

ROUGE-2 measures bigram overlap; captures more context \& coherence.

ROUGE-L measures longest common subsequence; captures fluency.

ROUGE-Lsum is a variant tailored for summaries.

Scores show reasonable abstractive summarization quality for T5-small on a reduced dataset subset.



### 4\. Example Outputs



Example 1:



Article:

Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films such as the 1956 noir While the City Sleeps died on March 15 at her home...



Reference Summary:

Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films died on March 15. Forrest, whose birth name was Katherine Feeney, had long battled cancer. A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films.



Predicted Summary:

Actress Sally Forrest was in the 1951 Ida Lupino-directed film 'Hard, Fast and Beautiful' and the 1956 Fritz Lang film 'While the City Sleeps'. Some of Forrest's other film credits included Bannerline, Son of Sinbad and Excuse My Dust.



### 5\. Analysis



Model captures key information from articles but may introduce factual discrepancies, common with T5-small.

Summaries are fluent and readable, with some variation in named entities and numbers.

Beam search with num\_beams=4 balances quality and generation speed.



### ✅ Conclusion:

The fine-tuned T5-small model achieves decent abstractive summarization with ROUGE-1 ≈ 41%, capturing key points of news articles. It’s ready for deployment and demo purposes.

