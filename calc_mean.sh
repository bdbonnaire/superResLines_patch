#!/bin/bash

for file in $*
do
	nb_exp=$(grep "offset_error_mean" $file | wc -l)
	nb_amp=$(grep "amp_error_mean" $file | wc -l)
	vals_offset=$(grep "offset_error_mean" $file | cut -d= -f2)
	vals_angle=$(grep "angle_error_mean" $file | cut -d= -f2)
	vals_amps=$(grep "amp_error_mean" $file | cut -d= -f2)
	vals_ampsLS=$(grep "ampLS_error_mean" $file | cut -d= -f2)

	sum_offset=0.
	sum_angle=0.
	sum_amps=0.
	sum_ampsLS=0.

	for i in $vals_amps
	do
		sum_amps=$(bc <<< "$sum_amps + $i")
	done
	for i in $vals_ampsLS
	do
		# echo $i
		sum_ampsLS=$(bc <<< "$sum_ampsLS + $i")
	done
	for i in $vals_angle
	do
		sum_angle=$(bc <<< "$sum_angle + $i")
	done
	for i in $vals_offset
	do
		sum_offset=$(python -c "$sum_offset + $i")
	done

	mean_offset=$(python -c "print($sum_offset / $nb_exp)")
	mean_angle=$(python -c "print($sum_angle / $nb_exp)")
	echo $sum_amps
	mean_amps=$(python -c "print($sum_amps / $nb_amp)")
	mean_ampsLS=$(python -c "print($sum_ampsLS / $nb_amp)")

	echo $file
	echo ----
	echo Amount : $nb_exp
	echo offset $mean_offset
	echo angle $mean_angle
	echo Amount Amps : $nb_amp
	echo amps $mean_amps
	echo ampsLS $mean_ampsLS
	echo
done
