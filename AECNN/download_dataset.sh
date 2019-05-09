#!/bin/bash


# specify the location to which the database be copied
DATADIR="./data"

# adapted from https://github.com/santi-pdp/segan
DATASETS="clean_trainset_wav noisy_trainset_wav clean_testset_wav noisy_testset_wav"

# DOWNLOAD THE DATASET
mkdir -p $DATADIR
pushd $DATADIR

for DSET in $DATASETS; do
    if [ ! -d ${DSET}_16kHz ]; then
        # Clean utterances
        if [ ! -f ${DSET}.zip ]; then
            echo 'DOWNLOADING $DSET'
            wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/${DSET}.zip
        fi
        if [ ! -d ${DSET} ]; then
            echo 'INFLATING ${DSET}...'
            unzip -q ${DSET}.zip -d $DSET
        fi
        if [ ! -d ${DSET}_16kHz ]; then
            echo 'CONVERTING WAVS TO 16K...'
            mkdir -p ${DSET}_16kHz
            pushd ${DSET}
            if [ ! $(ls *.wav) ]; then
		pushd ${DSET}
		ls *.wav | while read name
	    	do
                    sox $name -r 16k ../../${DSET}_16kHz/$name
		done
		popd
	    else
            ls *.wav | while read name
            do
                sox $name -r 16k ../${DSET}_16kHz/$name
            done
	    fi
            popd
        fi
    fi
done

popd

# make a copy of the filelists in datadir
#cp train_wav.txt test_wav.txt $DATADIR
