# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

for seed in 1 2 3 4 5
do
    echo "#!/bin/bash" > jobscript.sh
    echo "#SBATCH --time=7-00:00:00" >> jobscript.sh
    echo "#SBATCH --mem=10G" >> jobscript.sh
    echo "#SBATCH --gres=gpu:1" >> jobscript.sh
    echo "#SBATCH --nodelist=${1}" >> jobscript.sh
    echo "#SBATCH --job-name=${2}_seed_${seed}" >> jobscript.sh
    echo "#SBATCH --cpus-per-task=4" >> jobscript.sh
    echo "#SBATCH --ntasks=1" >> jobscript.sh
    echo "#SBATCH --output=output/%x-%j.out" >> jobscript.sh

    echo "source activate atom-ssl-ts" >> jobscript.sh
    echo 'echo "$(date): job $SLURM_JOBID starting on $SLURM_NODELIST"' >> jobscript.sh
    echo "hostname" >> jobscript.sh
    echo "whoami" >> jobscript.sh
    echo "nvidia-smi" >> jobscript.sh
    echo 'echo "CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES)"' >> jobscript.sh
    echo "# Print information about the branch and commit used." >> jobscript.sh
    echo 'echo -e "\nGit"' >> jobscript.sh
    echo 'echo -e "$(git branch -vv)"' >> jobscript.sh
    echo 'echo -e "current commit: $(git log --pretty=format:'%h' -n 1)  $(git log --pretty=format:'%H' -n 1)"' >> jobscript.sh

    echo "python -m ssl.run \
        --dataset \
            Chinatown \
            SonyAIBORobotSurface1 \
            ItalyPowerDemand \
            MoteStrain \
            SonyAIBORobotSurface2 \
            TwoLeadECG \
            SmoothSubspace \
            ECGFiveDays \
            Fungi \
            CBF \
            BME \
            UMD \
            DiatomSizeReduction \
            DodgerLoopWeekend \
            DodgerLoopGame \
            GunPoint \
            Coffee \
            FaceFour \
            FreezerSmallTrain \
            ArrowHead \
            ECG200 \
            Symbols \
            ShapeletSim \
            InsectEPGSmallTrain \
            BeetleFly \
            BirdChicken \
            ToeSegmentation1 \
            ToeSegmentation2 \
            Wine \
            Beef \
            Plane \
            OliveOil \
            SyntheticControl \
            PickupGestureWiimoteZ \
            ShakeGestureWiimoteZ \
            GunPointMaleVersusFemale \
            GunPointAgeSpan \
            GunPointOldVersusYoung \
            Lightning7 \
            DodgerLoopDay \
            PowerCons \
            FacesUCR \
            Meat \
            Trace \
            MelbournePedestrian \
            MiddlePhalanxTW \
            DistalPhalanxOutlineAgeGroup \
            MiddlePhalanxOutlineAgeGroup \
            ProximalPhalanxTW \
            ProximalPhalanxOutlineAgeGroup \
            DistalPhalanxTW \
            Herring \
            Car \
            InsectEPGRegularTrain \
            MedicalImages \
            Lightning2 \
            FreezerRegularTrain \
            Ham \
            MiddlePhalanxOutlineCorrect \
            DistalPhalanxOutlineCorrect \
            ProximalPhalanxOutlineCorrect \
            Mallat \
            InsectWingbeatSound \
            Rock \
            GesturePebbleZ1 \
            SwedishLeaf \
            CinCECGTorso \
            GesturePebbleZ2 \
            Adiac \
            ECG5000 \
            WordSynonyms \
            FaceAll \
            GestureMidAirD2 \
            GestureMidAirD3 \
            GestureMidAirD1 \
            ChlorineConcentration \
            HouseTwenty \
            Fish \
            OSULeaf \
            MixedShapesSmallTrain \
            CricketZ \
            CricketX \
            CricketY \
            FiftyWords \
            Yoga \
            TwoPatterns \
            PhalangesOutlinesCorrect \
            Strawberry \
            ACSF1 \
            AllGestureWiimoteY \
            AllGestureWiimoteX \
            AllGestureWiimoteZ \
            Wafer \
            WormsTwoClass \
            Worms \
            Earthquakes \
            Haptics \
            Computers \
            InlineSkate \
            PigArtPressure \
            PigCVP \
            PigAirwayPressure \
            Phoneme \
            ScreenType \
            LargeKitchenAppliances \
            SmallKitchenAppliances \
            RefrigerationDevices \
            UWaveGestureLibraryZ \
            UWaveGestureLibraryY \
            UWaveGestureLibraryX \
            ShapesAll \
            Crop \
            SemgHandGenderCh2 \
            EOGVerticalSignal \
            EOGHorizontalSignal \
            MixedShapesRegularTrain \
            SemgHandMovementCh2 \
            SemgHandSubjectCh2 \
            PLAID \
            UWaveGestureLibraryAll \
            ElectricDevices \
            EthanolLevel \
            StarLightCurves \
            NonInvasiveFetalECGThorax1 \
            NonInvasiveFetalECGThorax2 \
            FordA \
            FordB \
            HandOutlines \
        --run-name UCR \
        --loader UCR \
        --batch-size 8 \
        --repr-dims 320 \
        --max-threads 8 \
        --seed ${seed} \
        --queue-size 128 \
        --max-seq-len \
            24 \
            70 \
            24 \
            84 \
            65 \
            82 \
            15 \
            136 \
            201 \
            128 \
            128 \
            150 \
            345 \
            288 \
            288 \
            150 \
            286 \
            350 \
            301 \
            251 \
            96 \
            398 \
            500 \
            601 \
            512 \
            512 \
            277 \
            343 \
            234 \
            470 \
            144 \
            570 \
            60 \
            361 \
            385 \
            150 \
            150 \
            150 \
            319 \
            288 \
            144 \
            131 \
            448 \
            275 \
            24 \
            80 \
            80 \
            80 \
            80 \
            80 \
            80 \
            512 \
            577 \
            601 \
            99 \
            637 \
            301 \
            431 \
            80 \
            80 \
            80 \
            1024 \
            256 \
            2844 \
            455 \
            128 \
            1639 \
            455 \
            176 \
            140 \
            270 \
            131 \
            360 \
            360 \
            360 \
            166 \
            2000 \
            463 \
            427 \
            1024 \
            300 \
            300 \
            300 \
            270 \
            426 \
            128 \
            80 \
            235 \
            1460 \
            500 \
            500 \
            500 \
            152 \
            900 \
            900 \
            512 \
            1092 \
            720 \
            1882 \
            2000 \
            2000 \
            2000 \
            1024 \
            720 \
            720 \
            720 \
            720 \
            315 \
            315 \
            315 \
            512 \
            46 \
            1500 \
            1250 \
            1250 \
            1024 \
            1500 \
            1500 \
            1344 \
            945 \
            96 \
            1751 \
            1024 \
            750 \
            750 \
            500 \
            500 \
            2709 \
        --alpha 0.5 \
        --similarity INSTANCE TEMPORAL \
        --hierarchical False \
        --temperature 0.07 \
        --eval" >> jobscript.sh

    #submit the job
    sbatch jobscript.sh

    # remove the created script after submitting
    rm jobscript.sh
done
