# SparkMachineTranslation
Machine translation with Apache spark 


## Introduction 
The SparkMachineTranslation project aims to demonstrate the feasibility of large-scale machine translation using Apache Spark. This README provides an overview of the project's design details, focusing on two main tasks: Language Prediction and Language Translation.

## Design Details

In this project we explore the task of large scale machine translation.  The primary goal of this project is to show the feasibility of this challenge. We break this into two task. 
- Language Prediction 
- Languege Translation 


## Language detection 
Language prediction involves identifying the language of a given text. This task is essential for handling multilingual data efficiently. In this project, we utilize Apache Spark's distributed computing capabilities to analyze large volumes of text data. In many data processing applications bring data from different geographic regions to one compute resource for batch processing is a norm. Such data may contains different language text in the same field. i.e in a tabular structured data, same column might contain text from different languages. Our first goal is accurately detect the input text language and annotate those rows with appropriate tags. Then Find all the distinct tags and load the necessary translation models into memory and perform machine translation

## Language translation 
Language translation is the process of converting text from one language to another. In the context of this project, we leverage Apache Spark for scalable translation tasks. By distributing the translation workload across multiple nodes, we demonstrate the ability to handle translation tasks at a large scale effectively.



## 






```

```

