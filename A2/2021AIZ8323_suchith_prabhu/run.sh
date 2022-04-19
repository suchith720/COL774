if [ $# -gt 5 ] || [ $# -lt 4 ]
then
    echo ./run.sh 1 [path_to_train] [path_to_test] [part_num]
    echo ./run.sh 2 [path_to_train] [path_to_test] [binary_or_multi_class] [part_num]
    exit 1
fi


if ! [ -e $2 ]
then
    echo File does not exist, $1
    exit 1
fi
if ! [ -e $3 ]
then
    echo File does not exist, $2
    exit 1
fi

if [ $1 -eq 1 ]
then
    python 1.py $2 $3 $4
elif [ $1 -eq 2 ]
then
    if [ $4 == '0' ]
    then
        python 2_1.py $2 $3 $5
    elif [ $4 == '1' ]
    then
        python 2_2.py $2 $3 $5
    else
        echo Invalid option for binary_or_multi_class
    fi

else
    echo Invalid question number, $1
fi
