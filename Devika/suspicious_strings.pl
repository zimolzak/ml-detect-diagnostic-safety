#!/usr/bin/perl
# usage:
# cat *.ipynb | ./suspicious_strings.pl | less
while(<>){
    print if
	/[0-9]{5}/
	and length($_) < 2000
	and not /ORD_Singh_2019/
	and not /ML4TrgPos_Y201621/
	and not /Ttest_indResult/
	and not /Accuracy/
	and not /AUC/
	and not /F1/
}
