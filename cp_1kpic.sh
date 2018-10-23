index=0
train_num=1000
test_num=100
((cp_sum=$train_num+$test_num))
for file in /home/shenxj/RSNA/data/stage_1_train_jpg/*
do
	echo "$index"
	echo $file
	if [ $index -lt $train_num ];then
		cp $file /home/shenxj/RSNA/code/test/pic_train
	elif [ $index -lt $cp_sum ];then
		cp $file /home/shenxj/RSNA/code/test/pic_test
	else
		break
	fi
	((index++))	
done

echo train_num:$train_num
echo test_num:$test_num
echo cp_sum:$cp_sum
