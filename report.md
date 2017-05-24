# COMP90042 Project Report

Kuan Qian (Jack) [686464]
Zequn Ma []


## Introduction

In this project, we have made our attempts on building a question answering system. Starting with an implementation of the basic QA system described in the specification, we have made error analysis and introduced some enhancements to the system. The basic and enhanced system are then evaluated using several different metrics.

The question answering system takes 3 datasets in JSON format, for training, development and testing respectively. Each dataset is a JSON array, with each item being a "trial", that is, a set of documents(sentences) and questions. For training and development sets, answer to each question is also provided, including its content and label of the sentence containing the answer. The system takes the question along with a set of documents as input, and provides the answer as output.

## Basic QA System

For the first part of the project, a basic question answering system has been made. This does not only help us understanding the problem, but also provides a reasonable baseline evaluation for further development.

The basic system, as described in the specification, is formed by three main components. Firstly, the sentence retrieval component attempts to find the most relavant sentence to a certain question. The retrieved question is then used as a input for the entity extraction component, which aims to extract named entities in the sentence. The entities are finally ranked as potential answers to the question, with the top answer returned as the final result. Implementation details of the system is not included for brievty of the report, the system is designed and implemented to be consistent with the specification.

## Error Analysis

> - NER has significant room of improvement (only 24% identified in the correct (and retrieved) sentence, 29% in any retrieved sentence), among the failed cases:
> 	- About 70% that can possibly be extracted
>	- 10/35 partial match (measurements, symbols such as ‘,’ and ‘-’)
>	- Common words (noun phrases) not extracted
>	- Date not fully extracted
>	- Identifying more entities could possibly bring much more false positives, a good ranking algorithm will be required
> - Over 90% cases where correct sentence is retrieved in top 20, but only 60% as top 1. Consider improving sentence retrieval, or interpolate sentence ranking with other properties after rule 2
> - 16% when both of above are correct (correct sentence retrieved as top 1, in which correct answer is extracted by NER), but drops to 12% after applying answer ranking. With current implementation:
>	- Rule 1: No significant problem
>	- Rule 2: Question types not always correctly identified (88% accuracy for sample of 50 in training set, manually assessed), or entity type not correctly identified.
>	- Take sentence retrieval ranking: Some correct sentences not ranked as top 1, whose answers will be ranked lower to those in higher ranked sentences and passes Rule 1&2
>	- Rule 3: Not always helpful, sometimes prefers wrong answer. (stats?) All sorts of entities with type of ‘OTHER’, may confuse the ranker.


## Enhancement

> Rank answers using a classifier (from textbook, features at p8) FAILED
> Use ML classifier + NER for question type classification (along with more types)

## Evaluation

> Metrics: Exact match, partial match, MRR...
> Test for: Sentence retrieval, NER, answer ranking
> For rule 1/2/3, could be useful taking random as baseline



## Future Improvements

> - Take advantage of semantic information
> - Apply RNN for a “translation” from answers to questions (ref. TREC 2016 CMU paper)
> - Other approaches such as learnt pattern matching (ref. Learning Surface Text Patterns for a Question Answering System)


## Conclusion
