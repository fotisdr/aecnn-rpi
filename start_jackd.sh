#!/bin/bash

#Jackd starting script

# Jackd parameters - set your SOUNDDEVICE here!
SOUNDDEVICE=0 #RPiCirrus #audioinjectorpi
SOUNDSTREAM=0
FRAGSIZE=$1
SAMPLERATE=$2
NPERIODS=2

# kill unnecessary services
echo "killall jackd"
killall jackd -9 &> /dev/null
# uncomment the following lines to get better performance on a low-resource platform (e.g. a Raspberry Pi)
#sudo service ntp stop
#sudo service triggerhappy stop
#sudo killall console-kit-daemono
#sudo killall polkitd
#sudo mount -o remount,size=128M /dev/shm
#killall gvfsd
#killall dbus-daemon
#killall dbus-launch
#echo -n performance | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Initialization scripts necessary for the HAT sound card (Raspberry Pi)
# Cirrus audio card
#cd ~/bin
#./Record_from_Linein_Micbias.sh
#./Playback_to_Lineout.sh
# Audioinjector audio card
#alsactl --file /usr/share/doc/audioInjector/asound.state.MIC.thru.test restore

echo $'\nstart jackd'
#taskset -c 0
jackd --realtime -d alsa -d hw:$SOUNDDEVICE,$SOUNDSTREAM -p $FRAGSIZE -r $SAMPLERATE -n $NPERIODS -s 2>&1 | sed 's/^/[JACKD] /' &

sleep 2
