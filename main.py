import librosa
import os
import json

DATASET_PATH='dataset'
JSON_PATH='data.json'
SAMPLES_TO_CONSIDER=22050

def prepare_dataset(dataset_path,json_path,n_mfcc=13,hop_length=512,n_fft=2048):
    data={
        "mappings":[],
        "labels":[],
        "MFCCs":[],
        "files":[]
    }
    for i,(dirpath,dirname,filename) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            category=dirpath.split("/")[-1]
            data["mappings"].append(category)
            print(f'processing {category}')

            for f in filename:
                #get file path
                file_path=os.path.join(dirpath,f)

                #load audio file
                signal,sr=librosa.load(file_path)

                #ensure that the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    #enforse 1 sec long signal
                    signal=signal[:SAMPLES_TO_CONSIDER]

                    #extract the MFCCs
                    MFCCs=librosa.feature.mfcc(signal,n_mfcc=n_mfcc,hop_length=hop_length,
                                               n_fft=n_fft)

                    #sore data
                    data['labels'].append(i-1)
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f'{file_path}:{i-1}')
    #store in json file
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

if __name__=="__main__":
    prepare_dataset(DATASET_PATH,JSON_PATH)