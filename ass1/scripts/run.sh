#!/usr/bin/env bash

sizes=(500 1000 2000 4000 8000)
test_count=1
max_threads=$(($(nproc --all)))

mkdir -p log/pthread
for ((thread_count = 1; thread_count <= 16; thread_count *= 2)); do

    echo "Testing for $thread_count thread(s)"
    for size in ${sizes[*]}; do

        echo "on matrix of size $size"
        for ((i = 0; i < $test_count; i++)); do

            make gen size=$size > /dev/null
            output="$(make pthread num=$thread_count)"
            echo "$output" | grep "time" >> "./log/pthread/${size}_${thread_count}"

            if [[ $i == 0 ]]; then
                echo "$output" | grep -v "time" >> "./log/pthread/${size}_${thread_count}_out"
            fi

        done

    done

done



mkdir -p log/openmp
for ((thread_count = 1; thread_count <= 16; thread_count *= 2)); do

    echo "Testing for $thread_count thread(s)"
    for size in ${sizes[*]}; do

        echo "on matrix of size $size"
        for ((i = 0; i < $test_count; i++)); do

            make gen size=$size > /dev/null
            output="$(make openmp num=$thread_count)"
            echo "$output" | grep "time" >> "./log/openmp/${size}_${thread_count}"

            if [[ $i == 0 ]]; then
                echo "$output" | grep -v "time" >> "./log/openmp/${size}_${thread_count}_out"
            fi

        done

    done

done


mkdir -p log/sequential
for size in ${sizes[*]}; do

    echo "on matrix of size $size"
    for ((i = 0; i < $test_count; i++)); do

        make gen size=$size > /dev/null
        output="$(make sequential)"
        echo "$output" | grep "time" >> "./log/sequential/${size}"

        if [[ $i == 0 ]]; then
            echo "$output" | grep -v "time" >> "./log/sequential/${size}_out"
        fi

    done

done
