#!/bin/bash
rm -r ./output
mkdir output

#gcc -O3 -Wall -Werror -DDEBUG BMP_Processor_Serial_std.c -o BMP_Processor_Serial_std.out
exe="./Homework3"
for image in lady.bmp animal_small.bmp animal.bmp BackgroundRadiationMap.bmp photo3.bmp
do
  for filter in 1 2 3 4 5 6 7 8 9 10
  do
		#echo "perf stat -e "$value":u	 " $exe $n " >> results.txt 2>&1"
		#perf stat -e $value:u	$exe $n >> results.txt 2>&1 
		echo $exe $filter $image
		$exe $filter $image
  done
done

 