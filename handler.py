import json
import numpy as np
from scipy.sparse import lil_matrix
import boto3

client = boto3.client('sagemaker-runtime')

def serializer(data):
    js = { 'instances': [] }
    for row in data:
        js['instances'].append({ 'features': row.tolist() })
    return json.dumps(js)


def get_recommendation(event, context):
    samples = event['samples']

    X = lil_matrix((len(samples), 2625)).astype('float32')

    row = 0
    for sample in samples:
        user_id = int(sample['userId'])
        movie_id = int(sample['movieId'])

        user_index = user_id - 1
        movie_index = 943 + movie_id - 1
        X[row, user_index] = 1
        X[row, movie_index] = 1

        row = row + 1

    prediction = client.invoke_endpoint(
        EndpointName='factorization-machines-2019-06-03-17-08-52-679',
        Body=serializer(X.toarray()),
        ContentType='application/json'
    )

    responseBody = {
        "message": "Prediction successful!",
        "input": event,
        "predictions": json.loads(prediction['Body'].read().decode())['predictions']
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(responseBody)
    }

    return response