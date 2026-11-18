# End-to-end evaluation summary (32 questions)

Labels after manual review of every answer (`eval/manual_review.json`). `refused` = the system said it could not find the answer although the handbook contains it.

| Config | LLM | Prompt | Correct | Incomplete | Incorrect | Ambiguous | Refused | Out-of-scope declined | Mean latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | qwen2.5:0.5b | baseline | 12 | 1 | 18 | 1 | 0 | 0/2 | 10.7 |
| baseline-1.5b | qwen2.5:1.5b | baseline | 26 | 0 | 5 | 1 | 0 | 2/2 | 12.0 |
| grounded-v1-0.5b | qwen2.5:0.5b | v1 | 15 | 4 | 12 | 1 | 0 | 2/2 | 8.7 |
| grounded-v1-1.5b | qwen2.5:1.5b | v1 | 22 | 2 | 6 | 1 | 1 | 2/2 | 11.3 |
| grounded-v1-3b | qwen2.5:3b | v1 | 19 | 2 | 2 | 1 | 8 | 2/2 | 24.2 |
| improved-0.5b | qwen2.5:0.5b | v2 | 17 | 4 | 9 | 1 | 1 | 2/2 | 9.6 |
| improved-1.5b | qwen2.5:1.5b | v2 | 28 | 1 | 1 | 1 | 1 | 2/2 | 11.2 |
| improved-3b | qwen2.5:3b | v2 | 25 | 3 | 2 | 1 | 1 | 2/2 | 24.2 |
| improved-1.5b-c1000 | qwen2.5:1.5b | v2 | 25 | 2 | 3 | 1 | 1 | 2/2 | 13.0 |

## Automatic (keyword) labels before manual review

| Config | Correct | Incomplete | Incorrect | Refused | Mean fact recall |
|---|---|---|---|---|---|
| baseline | 19 | 3 | 10 | 0 | 0.65 |
| baseline-1.5b | 26 | 0 | 6 | 0 | 0.81 |
| grounded-v1-0.5b | 15 | 5 | 12 | 0 | 0.55 |
| grounded-v1-1.5b | 24 | 3 | 4 | 1 | 0.81 |
| grounded-v1-3b | 20 | 3 | 2 | 7 | 0.69 |
| improved-0.5b | 20 | 5 | 5 | 2 | 0.71 |
| improved-1.5b | 28 | 2 | 1 | 1 | 0.91 |
| improved-3b | 26 | 3 | 2 | 1 | 0.88 |
| improved-1.5b-c1000 | 26 | 2 | 3 | 1 | 0.85 |

## Per-question labels (manual)

| ID | Question | baseline | baseline-1.5b | grounded-v1-0.5b | grounded-v1-1.5b | grounded-v1-3b | improved-0.5b | improved-1.5b | improved-3b | improved-1.5b-c1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | What are the prerequisites for CS311 Advanced Deep Learning? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q02 | How many credits are required for the Computer Science major requirements? | incorrect | correct | incorrect | correct | correct | incorrect | correct | correct | correct |
| Q03 | How many credits and courses does the Computer Science minor require? | incorrect | correct | incomplete | incorrect | correct | incomplete | correct | correct | correct |
| Q04 | What must a student complete to formally declare Computer Science as their major? | incorrect | correct | incomplete | correct | correct | incorrect | correct | correct | correct |
| Q05 | What are the eligibility criteria for enrolling in CS 401 Capstone I? | incorrect | correct | correct | correct | correct | correct | correct | correct | correct |
| Q06 | What are the Major Learning Outcomes (MLOs) of the Computer Science program? | incorrect | correct | correct | correct | refused | correct | correct | correct | correct |
| Q07 | What are the Program Educational Objectives (PEOs) of the CS major? | correct | correct | incorrect | incomplete | refused | incomplete | incomplete | incomplete | incomplete |
| Q08 | What jobs have graduates of the B.S. Computer Science obtained? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q09 | Which graduate programs have Computer Science students applied to? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q10 | What is the main programming language used in CS201 Data Structures & Algorithms? | correct | correct | incorrect | correct | correct | correct | correct | correct | correct |
| Q11 | What are the three main parts of CS211 Operating Systems? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q12 | How often is CS103 Programming for Data Science and Visualization offered? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q13 | Is CS210 Computer Networks currently offered? | incorrect | incorrect | incorrect | incorrect | refused | incorrect | refused | refused | incorrect |
| Q14 | What are the prerequisites of CS302 Algorithms & Theory of Computing? | incorrect | correct | incorrect | correct | refused | incorrect | correct | correct | correct |
| Q15 | Which courses from other majors are cross-listed with Computer Science? | incorrect | correct | correct | correct | incomplete | correct | correct | correct | correct |
| Q16 | How many credits is a CS course worth unless indicated otherwise? | correct | correct | incorrect | correct | correct | correct | correct | correct | correct |
| Q17 | What are the four Core courses students must take? | incorrect | incorrect | incorrect | incorrect | refused | correct | correct | correct | incorrect |
| Q18 | How many exploratory courses are required, and how many of them can count toward the major? | incorrect | correct | incomplete | correct | refused | incomplete | correct | correct | correct |
| Q19 | What compulsory non-credit courses are required? | incorrect | correct | correct | refused | refused | refused | correct | incomplete | refused |
| Q20 | What options can a student take instead of Capstone I and Capstone II? | incorrect | correct | incorrect | incorrect | correct | incorrect | correct | correct | correct |
| Q21 | What kind of applications do students build in CS207 Software Construction? | correct | correct | incorrect | incorrect | incorrect | correct | correct | correct | correct |
| Q22 | Which assessment examples are used for PLO1 (fundamental computer science)? | incorrect | correct | incorrect | incorrect | incorrect | incorrect | incorrect | incorrect | incorrect |
| Q23 | Which universities' programs were used as references when building the CS curriculum? | correct | correct | incorrect | correct | correct | incorrect | correct | correct | correct |
| Q24 | What degree is awarded after completing the program requirements? | ambiguous | ambiguous | ambiguous | ambiguous | ambiguous | ambiguous | ambiguous | ambiguous | ambiguous |
| Q25 | List the sample advanced courses in the CS major. | incomplete | correct | incomplete | incomplete | incomplete | correct | correct | incomplete | incomplete |
| Q26 | What is the mission of Fulbright University Vietnam? | correct | correct | correct | correct | correct | correct | correct | correct | correct |
| Q27 | What is CS204 and what is its prerequisite? | correct | correct | correct | correct | correct | incomplete | correct | correct | correct |
| Q28 | In which year of the sample major plan do students take Discrete Math? | incorrect | incorrect | correct | correct | refused | correct | correct | incorrect | correct |
| Q29 | What is the tuition fee for the Computer Science major? | incorrect | correct | correct | correct | correct | correct | correct | correct | correct |
| Q30 | Who is the head of the Computer Science department? | incorrect | correct | correct | correct | correct | correct | correct | correct | correct |
| Q31 | What are its prerequisites? | incorrect | incorrect | incorrect | correct | correct | incorrect | correct | correct | correct |
| Q32 | And how do I formally declare it? | incorrect | incorrect | correct | correct | correct | incorrect | correct | correct | correct |

## Manual review notes

| Config | ID | Auto → manual | Note |
|---|---|---|---|
| baseline | Q01 | correct → correct | CS313 is right, but the description attributed to CS313 (GANs, RL, transformers) is actually CS311's |
| baseline | Q03 | incomplete → incorrect | claims the minor also requires a capstone course; CS201 and the 24 credits are missing |
| baseline | Q04 | correct → incorrect | adds an invented requirement of six courses, including a non-existent CS 202 |
| baseline | Q05 | correct → incorrect | GPA 3.5 + application are right, but four extra criteria are invented |
| baseline | Q06 | correct → incorrect | adds a fourth 'MLO' copied from the program description |
| baseline | Q14 | correct → incorrect | garbles the prerequisites ('CS201: Computer Science (CS101)', 'MATH 202: Mathematics (CS202)') |
| baseline | Q18 | incomplete → incorrect | 'up to 2 exploratory courses (24 credits)'; never says that 6 are required |
| baseline | Q22 | correct → incorrect | lists teaching methods as assessment examples and invents course-code mappings |
| baseline | Q27 | correct → correct | title and prerequisite right; adds invented topics (pattern recognition) |
| baseline | Q30 | correct → incorrect | speculates that the head is 'a professor or faculty member' |
| baseline | Q32 | correct → incorrect | answers the rule for declaring the major (two courses) instead of the minor |
| baseline-1.5b | Q02 | correct → correct | 56 credits is right; the breakdown lists the OR-alternatives as if cumulative |
| grounded-v1-0.5b | Q07 | incomplete → incorrect | second 'PEO' is a sentence from the program description |
| grounded-v1-1.5b | Q03 | incomplete → incorrect | '6 credits, 6 courses' - the minor is 24 credits |
| grounded-v1-1.5b | Q20 | correct → incorrect | 'Four additional intermediate courses' instead of two |
| grounded-v1-1.5b | Q22 | correct → incorrect | merges the teaching-methods column into the assessment examples |
| grounded-v1-3b | Q12 | correct → correct | right, but the answer starts with a copied passage header '(CS_MajorHandbook.pdf, p. 2)' |
| grounded-v1-3b | Q16 | correct → correct | copies the raw passage '[3] (CS_MajorHandbook.pdf, p. 13) ... (4 credits)' instead of answering |
| grounded-v1-3b | Q18 | incomplete → refused | answered 'I could not find this information' although the passage was retrieved |
| grounded-v1-3b | Q22 | correct → incorrect | merges the teaching-methods column into the assessment examples |
| improved-0.5b | Q04 | incomplete → incorrect | claims the two courses must be CS101 and CS103 |
| improved-0.5b | Q14 | correct → incorrect | lists CS302 itself as one of its own prerequisites |
| improved-0.5b | Q20 | refused → incorrect | degenerate list 'CS 401 ... CS 412: Capstone Project' (cut by the 512-token cap) |
| improved-0.5b | Q22 | correct → incorrect | describes teaching methods (Lecture, Lab) as assessment examples |
| improved-0.5b | Q25 | correct → correct | all 15 courses, then starts repeating items |
| improved-0.5b | Q32 | correct → incorrect | invents requirements (CS201, three intermediate courses, capstone) |
| improved-1.5b | Q03 | incomplete → correct | lists all six courses and 24 credits (keyword 'six' missing, so the auto-label was incomplete) |
| improved-1.5b | Q22 | correct → incorrect | merges the teaching-methods column (Lecture, Lab, ...) into the assessment examples |
| improved-1.5b-c1000 | Q22 | correct → incorrect | merges the teaching-methods column (Lecture, Lab, ...) into the assessment examples |
| improved-3b | Q15 | correct → correct | all three courses, but copies the table cells 'CS TBA' after each title |
| improved-3b | Q22 | correct → incorrect | merges the teaching-methods column (Lecture, Lab, ...) into the assessment examples |
