#required libraries
import urllib.request
import scipy.io.wavfile
import pydub
import numpy as np
import matplotlib.pyplot as plt 
import os
import io
import librosa
import librosa.display
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from google.cloud import storage
import json
import requests

# POST
AUTH_URL = 'https://accounts.spotify.com/api/token'
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': "",
    'client_secret': "",
})
# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

#For Google Cloud Bucket
def upload_blob(bucket_name, source_file_name, destination_blob_name):
	"""Uploads a file to the bucket."""
	# The ID of your GCS bucket
	# bucket_name = "your-bucket-name"
	# The path to your file to upload
	# source_file_name = "local/path/to/file"
	# The ID of your GCS object
	# destination_blob_name = "storage-object-name"

	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(destination_blob_name)

	blob.upload_from_filename(source_file_name)

	print(
		"File {} uploaded to {}.".format(
			source_file_name, destination_blob_name
		)
	)

# DELETE
def delete_blob(bucket_name, blob_name):
	"""Deletes a blob from the bucket."""
	# bucket_name = "your-bucket-name"
	# blob_name = "your-object-name"

	storage_client = storage.Client()

	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(blob_name)
	blob.delete()

	print("Blob {} deleted.".format(blob_name))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'geometric-shore.json'
storage_client = storage.Client()
bucket = storage_client.get_bucket('cs467')

def uploadToCloud(url, genre, count):
    webFile = url
    number = str(count)
    while len(number) < 5:
        number = '0' + number

    name = "{}{}.{}.mp3".format('temp/', genre, number)
    blob = "{}{}.{}.wav".format('temp/', genre, number)
    try:
        urllib.request.urlretrieve(webFile, name)
        sound = pydub.AudioSegment.from_mp3(name)
        sound.export(blob, format='wav')
        #read mp3 file
        #mp3 = pydub.AudioSegment.from_mp3(name)
        #blob = mp3.export(name, format="wav")
        destinationName = "{}{}.{}.wav".format('dddM1000/', genre, number)
        upload_blob('cs467', blob, destinationName)
    except:
        print("failed to download skipped.")
    #delete the temp download.
    try:
        os.remove(blob)
        os.remove(name)
    except:
        print("unable to remove file {}".format(name))

if __name__ == "__main__":
    max_offset=10
    runTrack = 0
    collection = 0
    maxCollection = 1000#10 #999

    #open text file to write stuff
    file1 = open("songList.txt", "w")
    collectionDict = {
        #'classical': 0,  
        #'country': 0,   
        #'eletronic': 0, 
        #'folk': 0, 
        #'Hip-Hop': 0, 
        #'jazz': 0, 
        #'blues': 0, 
        #'pop': 0, 
        #'rock': 0, 
        #'disco': 0, 
        #'metal': 0, 
        #'reggae': 0 
    }
    # The following searches top artists in each genre to a max limit
    # The code checks to download the exact genre first then it will download any
    # subgenres for each genre of music. 
    # then it will search artists top 20 songs that contain
    # preview_url; which can be downloaded and uploaded to the cloud.
    # The loop will break once 1000 songs for each genre is collected.

    songData = collectionDict.copy()
    #extraList = collectionDict.copy()
    for gTypes in collectionDict:
        count = 0
        print(gTypes)
        genre=gTypes
        artList = {}
        extraList = artList.copy()
        idx = 0
        start = 0
        failsafe = 0
        idx = 0
        while start <= 100 and failsafe < 10:
            print(start, "failsafe is =", failsafe)
            try:
                #results = sp.search(q='genre:rock', offsetex=start, limit=50)
                #start += 50
                failsafe += 1
                r=requests.get('https://api.spotify.com/v1/search?q=genre:{} NOT reggaeton&type=artist&offset={}&limit=50'.format(genre, start), headers=headers)
                artists = r.json()
            except:
                start += 10
                failsafe += 1
                continue
            
            try:
                flag = artists['artists']['items']
            except:
                #no more in list move on to the next one
                failsafe += 1
                continue
            if len(artList) + len(extraList) >= 100:
                break

            for artistsIdx, artist in enumerate(artists['artists']['items']):
                if len(artList) + len(extraList) >= 100:
                    break
                print(artist['name'], artist['genres'])
                if gTypes in artist['genres']:
                    #print("Gotcha")
                    #alist.append({idx, gTypes, artist['genres'], artist['name']})
                    #print(idx, gTypes, artist['genres'], artist['name'])
                    if artist['name'] in artList:
                        #already have name
                        print("artist already in artList")
                    else:
                        artList[artist['name']] = artist['genres']
                #elif gTypes == "electronic":
                else:
                    #this means that the artist does not contain main songs genre
                    if artist['name'] in extraList:
                        #already have name
                        print("artist already in artList")
                    else:
                        extraList[artist['name']] = artist['genres']
                collectionDict[gTypes] += 1
                start += 1

                if len(artList) + len(extraList) >= 100:
                    break
        #iterate throug the artists and get their top songs
        idx = 0
        songList = []
        for artName in artList:
            if len(songList) >= maxCollection:
                break

            print(idx, artName, artList[artName])
            idx += 1
            #Search by artist and get tracks by them
            #href": "https://api.spotify.com/v1/search?query=tania+bowra&offset=0&limit=20&type=artist"
            try:
                #results = sp.search(q='genre:rock', offset=start, limit=50)
                r=requests.get("https://api.spotify.com/v1/search?query=artist:{}&offset=0&limit=20&type=track".format(artName), headers=headers)
                tracks = r.json()
            except:
                continue
            
            songCount = 0
            for track in tracks:
                if len(songList) >= maxCollection:
                    break
                #print(tracks[bob])
                try:
                    for x in tracks[track]['items']:
                        if len(songList) >= maxCollection:
                            break
                        #print(songCount, x["href"])
                        try:
                            #results = sp.search(q='genre:rock', offset=start, limit=50)
                            r=requests.get(x["href"], headers=headers)
                            payload = r.json()
                        except:
                            continue
                        if len(songList) >= maxCollection:
                            break

                        print(idx, artName, payload['name'])
                        #print(songCount, tracks["preview_url"])
                        if payload["preview_url"] != None:
                            songList.append(payload["preview_url"])

                except:
                    continue
            
                if len(songList) >= maxCollection:
                    break
    
        for artName in extraList:
            if len(songList) >= maxCollection:
                break

            print(idx, artName, extraList[artName])
            idx += 1
            #Search by artist and get tracks by them
            #href": "https://api.spotify.com/v1/search?query=tania+bowra&offset=0&limit=20&type=artist"
            try:
                #results = sp.search(q='genre:rock', offset=start, limit=50)
                r=requests.get("https://api.spotify.com/v1/search?query=artist:{}&offset=0&limit=20&type=track".format(artName), headers=headers)
                tracks = r.json()
            except:
                continue
            
            #print(tracks)
            songCount = 0
            for track in tracks:
                if len(songList) >= maxCollection:
                    break
                #print(tracks[bob])
                try:
                    for x in tracks[track]['items']:
                        if len(songList) >= maxCollection:
                            break
                        #print(songCount, x["href"])
                        try:
                            #results = sp.search(q='genre:rock', offset=start, limit=50)
                            r=requests.get(x["href"], headers=headers)
                            payload = r.json()
                        except:
                            continue
                        if len(songList) >= maxCollection:
                            break

                        print(idx, artName, payload['name'])
                        #print(songCount, tracks["preview_url"])
                        if payload["preview_url"] != None:
                            songList.append(payload["preview_url"])
                            file1.write("{} {} = {} -  {} \n".format(idx, artName, payload['name'], payload["preview_url"]))
                except:
                    continue
            
                if len(songList) >= maxCollection:
                    break
        songIdx = 0
        for i in songList:
            print(str(songIdx + 1), gTypes, "DOWNLOAD SONG", i)
            songIdx += 1
            #uploadToCloud(i, 'hiphop', songIdx)
        #songData[gTypes] = songList
    file1.close()
    print("Collection count is {}".format(collectionDict))
    #print("Song count is {}".format(songData))
