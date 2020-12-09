import json
import boto3
import os
from sagemaker.mxnet.model import MXNetPredictor
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

s3_client = boto3.resource('s3')
ses_client = boto3.client('ses')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))
    
    for each_email in event['Records']:
        bucket = each_email['s3']['bucket']['name']
        key = each_email['s3']['object']['key']
        
        # retrieve information part
        email_obj = s3_client.Object(bucket, key)
        email_file_body = str(email_obj.get()['Body'].read())
        Info_start = email_file_body.rfind("From:")
        email_file_body = email_file_body[Info_start:]
        
        
        # get email body
        CLASSIFICATION = ''
        CLASSIFICATION_CONFIDENCE_SCORE = 0.00
        
        email_body = email_file_body.replace("\\r", "").replace("\\n", "")
        left_charset = email_body.find("charset=")
        right_charset = email_body.rfind("charset=")
        email_body = email_body[left_charset:right_charset]
        right = email_body.rfind("--0000")
        email_body = email_body[15:right]
        if len(email_body) > 240:
            email_body = email_body[:240]
            
            
        # get the classification result of email_body
        mxnet_pred = MXNetPredictor('spam')
        vocabulary_length = 9013
        test_messages = [email_body]
        one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

        result = mxnet_pred.predict(encoded_test_messages)
        CLASSIFICATION_CONFIDENCE_SCORE = round(float(result["predicted_probability"][0][0]) * 100, 2)
        if result["predicted_label"][0][0] == 1.0:
            CLASSIFICATION = "Spam Message"
        elif result["predicted_label"][0][0] == 0.0:
            CLASSIFICATION = "Ham Message"
        
        # get other email information
        email_info = email_file_body.split("\\r\\n")
        email_address = ''
        email_date = ''
        email_subject = ''
        
        for each_info in email_info:
            if each_info.startswith('From: '):
                starting_pos = each_info.find('<') + 1
                email_address = each_info[starting_pos:-1]
                
            elif each_info.startswith('Date: '):
                starting_pos = each_info.find(':') + 2
                temp = each_info[starting_pos:]
                temp_end = temp.find(':') - 2
                email_date = temp[:temp_end].strip()
                
            elif each_info.startswith('Subject: '):
                starting_pos = each_info.find(':') + 2
                email_subject = each_info[starting_pos:]
        
        reply1 = 'We received your email sent at ' + email_date + ' with the subject ' + email_subject + '.\n'
        reply2 = 'Here is a 240 character sample of the email body: \n'
        reply3 = email_body + '\n'
        reply4 = 'The email was categorized as ' + CLASSIFICATION + ' with a ' + str(CLASSIFICATION_CONFIDENCE_SCORE) + '% confidence.'
        
        response = ses_client.send_email(
            Source = 'test@proj4.cc.qli5.top',
            Destination = {
                'ToAddresses': [
                    email_address,
                ]
            },
            Message = {
                'Subject': {
                    'Data': email_subject
                },
                'Body': {
                    'Text': {
                        'Data': reply1 + reply2 + reply3 + reply4
                    },
                    'Html': {
                        'Data': '<p>' + reply1 + '</p>' + '<p>' + reply2 + '</p>' + '<p>' + reply3 + '</p>' + '<p>' + reply4 + '</p>'
                    }
                }
            },
            ReplyToAddresses = [
                email_address,
            ]
        )
        
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
