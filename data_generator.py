import cv2
import os
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm

SEED = 10
random.seed(SEED)
np.random.seed(SEED)

# Dataset Parameters
img_size = 224
size = 15
question_size = 10 # 6 for one-hot vector of color, 1 for question type, 3 for question subtype
q_type_idx = 6
sub_q_type_idx = 7
nb_questions = 10
# Possibles Answers : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]

colors = [
    (0,0,255), # red
    (0,255,0), # green
    (255,0,0), # blue
    (0,156,255), # orange
    (128,128,128), # gray
    (0,255,255) # yellow
]

def center_generate(objects):
    '''Generate centers of objects'''
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name, c, shape, _ in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_sample():
    '''Returns an image (with its bbox and attributes) and the corresponding questions, programs and answers'''
    
    # Create objects
    objects = [] # [(color, center, shape, xmin, ymin, xmax, ymax), (...), ...]
    img = np.ones((img_size,img_size,3)) * 255
    for color_id, color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id, center, 'r', (start[0], start[1], end[0], end[1]))) 
        else:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id, center, 'c', (start[0], start[1], end[0], end[1])))


    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    
    # Non-Relational Questions
    for _ in range(nb_questions):
        
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 0
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        
        if subtype == 0:
            # query shape -> rectangle/circle
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            # query is left side (horizontal position) -> yes/no
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            # query is up side (vertical position) -> yes/no
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    # Relational Questions
    for _ in range(nb_questions):
        
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        rel_questions.append(question)

        if subtype == 0:
            # closest to -> rectangle/circle
            my_obj = objects[color][1]
            distances = np.array([np.linalg.norm(np.array(my_obj) - np.array(obj[1])) for obj in objects])
            sorted_dists = distances.argsort()
            idx = sorted_dists[0] if distances[sorted_dists[0]] != 0 else sorted_dists[1]
            if objects[idx][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            # furthest from -> rectangle/circle
            my_obj = objects[color][1]
            distances = np.array([np.linalg.norm(np.array(my_obj) - np.array(obj[1])) for obj in objects])
            sorted_dists = distances.argsort()
            idx = sorted_dists[-1] if distances[sorted_dists[-1]] != 0 else sorted_dists[-2]
            if objects[idx][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            # count -> 1~6
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count += 1 
            answer = count + 4

        rel_answers.append(answer)

    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)
    img = img / 255.
    sample = (img, objects, relations, norelations)
    
    return sample

def convert_sample(sample):
    # Convert vector representation to Natural language questions and programs

    img, objects, (rel_questions, rel_answers), (norel_questions, norel_answers) = sample
    colors = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers

    queries = []
    programs = []
    text_answers = []

    for i, (question, answer) in enumerate(zip(questions, answers)):
        query = f'Q{i}. '
        color = colors[question.tolist()[0:6].index(1)]

        # Non relational questions
        if question[q_type_idx] == 0:
            if question[sub_q_type_idx] == 1:
                queries.append(f'What is the shape of the {color} of the object?')
                programs.append(f'filter {color} <nxt> query shape')

            elif question[sub_q_type_idx + 1] == 1:
                queries.append(f'Is there a {color} object on the left?')
                programs.append(f'filter {color} <nxt> query positions <nxt> isLeft')

            else:
                queries.append(f'Is there a {color} object on the top?')
                programs.append(f'filter {color} <nxt> query position <nxt> isTop')
            
        # Relational questions
        else:
            if question[sub_q_type_idx] == 1:
                queries.append(f'What is the shape of the closest object to the {color} object?')
                programs.append(f'filter {color} <nxt> relate closest <nxt> query shape')

            elif question[sub_q_type_idx + 1] == 1:
                queries.append(f'What is the shape of the farthest object to the {color} object?')
                programs.append(f'filter {color} <nxt> relate farthest <nxt> query shape')

            else:
                queries.append(f'How many objects of the shape as that of the {color} are there?')
                programs.append(f'filter {color} <nxt> query shape <nxt> filter <nxt> count')


        ans = answer_sheet[answer]
        text_answers.append(ans)

    return img, objects, queries, text_answers, programs


def build_dataset(num_samples, data_dir, prefix):
    '''Builds a Full Dataset with Images, Detector Data, Attribute Data, Queries, Programs and Answers'''
    samples = [build_sample() for _ in range(num_samples)]

    # Init dataframes 
    img_det_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    que_to_prog = pd.DataFrame(columns=['answer', 'query_text', 'program_text'])
    img_dir = os.path.join(data_dir, 'images')

    shape_map = {'r':'rectangle', 'c':'circle'}
    
    try:
        os.makedirs(data_dir)
    except:
        pass

    try:
        os.makedirs(img_dir)
    except:
        pass

    for i, sample in enumerate(tqdm(samples)):
        # Get Data
        img, objects, queries, programs, answers = convert_sample(sample)
        file_name = f'{i}.jpg'
        img_path = os.path.join(img_dir, file_name)
        cv2.imwrite(img_path, img * 255)

        for obj in objects:
            # get object parameters
            color_id, shape, bbox = obj[0], shape_map[obj[2]], obj[3]

            img_det_df = pd.concat([img_det_df, pd.DataFrame([{'filename': file_name, 
                                                    'width': img_size, 
                                                    'height': img_size, 
                                                    'class': 'obj', 
                                                    'xmin': bbox[0], 'ymin': bbox[1],
                                                    'xmax': bbox[2], 'ymax': bbox[3]}])], ignore_index=True)
        
        # Append text data to dataframe
        for answer, query, program in zip(answers, queries, programs):
            que_to_prog = pd.concat([que_to_prog, pd.DataFrame([{'filename': file_name,
                                              'answer': answer,
                                              'query_text': query,
                                              'program_text': program}])], ignore_index=True)

        img_det_df.to_csv(os.path.join(data_dir, f'{prefix}_img_det.csv'), index=False)
        que_to_prog.to_csv(os.path.join(data_dir, f'{prefix}_que_to_prog.csv'), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset genertor')
    parser.add_argument('--n_train', type=int, default=50, help='Number of training images to be generated')
    parser.add_argument('--n_test', type=int, default=100, help='Number of testing images to be generated')
    args = parser.parse_args()

    print('Generating training images...')
    build_dataset(args.n_train, 'train', 'train')
    
    print('Generating test images...')
    build_dataset(args.n_test, 'test', 'test')
