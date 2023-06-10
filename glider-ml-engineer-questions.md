# Machine Learning Engineer questions 

## Question 1:
### Topic: AWS Kinesis
What is the buffer size and buffer interval on Kinesis Data Firehose which buffers incoming
streaming data to a certain size or for a certain period of time before delivering it to destinations?

A - Buffer Size is in bytes and Buffer Interval is in seconds.

B - Buffer Size is in MBs and Buffer Interval is in seconds.

C - Buffer Size is in MBs and Buffer Interval is in minutes

D - Buffer Size is in GBs and Buffer Interval is in minutes.

E - Buffer Size is in GBs and Buffer interval is in seconds.

**Answer**: B

**Justification**:
In Amazon Kinesis Data Firehose, the buffer size is specified in megabytes (MB), and the buffer interval is specified in seconds. This allows you to configure how much data should be accumulated (buffered) before it is delivered to the destination.

By setting the buffer size, you determine the maximum amount of data that will be buffered before triggering a delivery to the destination. The buffer interval defines the maximum time duration for which data can be buffered before it is delivered.
## Question 2:
### Topic: Tensorflow
Which kind of neural network architecture is commonly used for object estimation is na autonomous self-driving car:

A- Multilayer perceptron

B- CNN

C- Simple Recurrent Neural Network

D- Deep Belief Network

E- Generative Adversarial Network

**Answer**: B

**Justification**:
CNNs are widely used in computer vision tasks, including object detection and estimation. They excel at extracting meaningful features from visual data, making them suitable for analyzing images and video frames. CNNs leverage convolutional layers to automatically learn hierarchical representations of visual features, enabling them to effectively detect and classify objects in complex scenes.

While other neural network architectures like multilayer perceptrons (a), simple recurrent neural networks (c), deep belief networks (d), and generative adversarial networks (e) have their applications in different domains, CNNs have become the go-to choice for object estimation in autonomous self-driving cars due to their ability to process and analyze visual data efficiently.
## Question 3:
### Topic: Tensorflow
The independent and dependent features of data are represented, as X and y respectively. These are further split into
training and testing sets, named as X_train, y_train, X_test, y_test. Complete the folowing
code for optimization with gradient Tape.

```python
with tf.Gradient_Tape() as g:
    pred = lstm_net()
    
    loss = cross_entropy(y_train, pred)

```
**Answer** Not sure: 1-X_train
2-(y_train, pred)


## Question 4:
### Topic: Tensorflow

Rearrange the following in the correct sequence of the working of tensorflow algorithms.

1-Import and pre-process data

2-Employ computational graphs for data input

3- Modify variables using the backpropagation algorithm

4- Evaluate output on the loss function

5- Repeat till termination are criteria met 

**Answer and justification**
1- Import and pre-process data: This step involves importing the required data into your TensorFlow program and performing any necessary pre-processing steps, such as data cleaning, normalization, or feature engineering.

2- Employ computational graphs for data input: In TensorFlow, you define your computations using computational graphs. This step involves creating placeholders or input layers to represent the data input to your model. These placeholders serve as the entry points for your data during the training or inference process.

3- Modify variables using the backpropagation algorithm: TensorFlow uses automatic differentiation and the backpropagation algorithm to compute gradients and update the model's variables during training. This step involves defining your model's architecture and operations, calculating the loss based on the model's predictions and the true labels, and using the gradients to update the model's variables.

4- Evaluate output on the loss function: Once the variables are updated, you evaluate the output of your model on the loss function. The loss function measures the discrepancy between the predicted output and the true output. This evaluation helps you assess the performance of your model and guide the optimization process.

5- Repeat till termination criteria met: The training process is typically performed iteratively until a termination criterion is met. This criterion can be a maximum number of iterations, reaching a desired level of accuracy, or observing a plateau in the model's performance. The process involves repeatedly executing steps 2 to 4, updating the model's variables, evaluating the loss, and adjusting the model's parameters until the termination criteria are satisfied.

## Question 5:
### Topic: PyTorch
You have a tensor x defined as follows:
1. x = torch. FloatTensor( [ I10, 20, 30), [30, 40, 50], [60, 70, 80]])
**Part A**
We want to compute the sum of the on-diagonal elements. Identify the correct option to do this.

A-  1. torch. trace (x)

B-  1. torch.s um (x [: :-1] )

C-  1. torch.diagonal_sum(x)

D   1. x.diagonal_sum(x)

**Answer** A

**Part B**

You want to swap the 1st dimension of x with the 2nd dimension. Identify the correct option to do
this.

A-  1. X.permute 1, 0))

B-  1. x.transpose((1, 0))

C-  1. x.permute((0, 1))

D-  1. x.transpose( (2, 0))

E-  1. x.T

**Answer** B
## Question 6:
### Topic: PyTorch
You want to do inference using a trained model. The model object is net.

**Part A**

How do you make sure that the computation graph does not store anything for gradient
computations?

A- net.no_grad)

B- torch.no_grad()

C- net.eval()

D- net(no_grad)

E- net.eval

**Answer** B

**Justification**:
In PyTorch, you can use the torch.no_grad() context manager to disable gradient tracking. This is useful when performing inference or evaluation using a trained model and you don't need to compute gradients. By wrapping your inference code within torch.no_grad(), the computation graph will not store any operations for gradient computations, resulting in reduced memory consumption and improved performance during inference.

**Part B**

How will you turn off dropout layers during inference?

A- net.no_grad()

B- net.eval()

C- torch.no _grad()

D- torch.eval()

E- torch.eval(net)

**Answer** B

**Justification**
In PyTorch, calling net.eval() on a model switches it to evaluation mode. This mode turns off the dropout layers, which are commonly used for regularization during training. During inference, you typically want to disable dropout because you want the model to make predictions based on the full capacity of the network.

## Question 7:

### Topic: SparksQL
Choose twO reasons for Spark to give an Out of Memory Error in a spark SQL execution
environment.(Select all that apply)

A- Explain plan lineage

B- Yarn Memory Overhead

C- Low concurrency operation

D- Broadcast Join on a small data frame

E Collect Operation called ona large data frame

**Answer** B, E

## Question 8:

A sql interpreter and optimizer handles the functional programming of spark sql and transform the data frame RDD to get the required results in the desired format. select the relevant features associated with it:

A- streaming uncatalyst

B- Process all size of data

C- Faster than RDDs

D-transform trees

E-functional programming

**Answer** C, D, E

**Justification**
c) Faster than RDDs: Spark SQL and its optimizer are designed to provide faster query execution compared to RDD-based operations. By leveraging advanced optimizations and query planning techniques, Spark SQL can efficiently process and analyze large datasets.

d) Transform trees: In the context of Spark SQL, transform trees refer to the process of applying various transformations and optimizations to the query execution plan. The SQL interpreter and optimizer analyze the query, perform transformations on the query plan, and optimize it for efficient execution. This involves restructuring and optimizing the logical and physical execution plans using techniques such as predicate pushdown, join reordering, and column pruning.

e) Functional programming: Spark SQL incorporates functional programming concepts by providing a DataFrame API that allows developers to perform transformations and aggregations on distributed datasets in a functional style. With functional programming, data transformations are expressed as a series of operations on immutable DataFrames, promoting ease of use and enabling a more declarative and expressive coding style.


## Question 9:
### Topic: Spark sQL
Consider the below data frame:

1. emp = [('A','IT'no-reply@abc.edu', 10000), ('C', 'IT', 'no-reply@abc. edu', 25000),
('B', 'FINANCE' n0-reply@abc.edu, 15000), ((D', 'FINANCE')'no-reply@abc. edu', 25000)
('E', 'OPS ,no-reply@abc.edu, 20000), ('F', '0PS', 'n0-reply@abc.edu ', 15000)]

df = sqlContext.createDataFrame(emp, ["firstName", "dep'","email", "salary" ])

How to add a new column [factor] similar to as shown below?

|firstName |dep|       email             |salary | factor
A           IT         n0-reply@abc.edu   10000    0.8
C           IT         n0-reply@abc.edu   25000    0.8
B           FINANCE    n0-reply@abc.edu   15000    0.01
D           FINANCE    n0-reply@abc.edu   25000    0.01
E           OPS        n0-reply@abc.edu   20000    0.6
F           OPS        n0-reply@abc.edu   15000    0.6

A- df.with Column('factor',F.when(df.dep == 'IT', 0.8),when(df.dep =='FINANCE', 0.01).otherwise(0.6)).show()

B- sqlcontext.sql('select factor from df where df.dep == 'IT' and df.dep == 'FINANCE' and df.dep =='OPS ).show()

C- df('factor,F.when(df.dep == 'IT', 0.8).when(df.dep =='FINANCE, 0.01)).show()

D- df.sparksql(F.when(df.dep =='IT', 0.8).when(df.dep =='FINANCE', 0.01).otherwise(0.6)).show()

E- df.sql(factor, E.when(df.dep ==IT, 0.8).when(df.dep =='FINANCE, 0.01)).show()

**Answer** A

## Question 10:

### Feature Scaling - Normalization Method
You have an Employee Data Frame with details of employees as below.

Employee Dataframe structure:

id blood_group age salary

id: Integer
blood_group: String
age: Integer
salary: Float

Your task is to drop employee details if there are no
entries for salary. Perform feature scaling on age
and salary columns using Max-Min Normalization
method required for further machine learning proces.

**Input**

df = pd.DataFrame({'id':[0,1,2,3], 'blood_group':[A,A,A,B], 'age':[30,55,61,63], 'salary':[2297.0,1134.0,4969.0,]})

**Output**
df = pd.DataFrame({'id':[0,1,2], 'blood_group':[A,A,A], 'age':[30,55,61], 'salary':[2297.0,1134.0,4969.0]})

**Explanation**
Here, id-3 was dropped since it has no entry for salary and the values of age and salary columns are normalized.

**Note**: The output dataframe should contain only the four columns mentioned above and the normalized values should be rounded off to four decimal points.

```python
#1/bin/python3
import re
import sys
import math
import random
import os
import pandas as pd
import sys
import io
from datetime import datetime
import numpy as np

def load_data():
    # data is the dataset stores the listing information
    headers = ['id', 'blood group', 'age', 'salary']
    input_stream = io.TextIOWrapper(sys. stdin. buffer, encoding='utf-8')
    data = pd.read_csv(input_stream, encoding = 'utf-8', sep= ',')
    return data


def normScaling(data, X):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    # Add your code here to perform normalize scaling operation on dataframe columr
    scaler = MinMaxScaler()
    column_values = data[X].values.reshape(-1, 1)
    # Scale the column using fit_transform
    scaled_values = scaler.fit_transform(column_values)
    scaled_values = np.round(scaled_values, 4)
    data[X] = scaled_values.flatten()
    return data


def get_result():

    data = load_data()


    # Add your code here drop the missing value rows & apply the nomScal ing function
    # Drop rows where 'salary' column has missing values (NaN)
    data.dropna(subset=['salary'], inplace=True)
    columns = ['salary', 'age']
    for col in columns:
        data[col] = data[col].astype('float64')
        data= normScaling(data, col)
    
    return data;


if __name__ =='__main__':
    outcome = get_result()
    out_stream = io.TextIOWrapper(sys.stdout . buffer, encoding='utf-8')
    outcome.to_csv(out_stream, encoding = 'utf-8', sep= ',', index=False)


```