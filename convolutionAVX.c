#define N3
#ifdef N3


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include "image_types.h"
#include "handle_image_header.h"
#include "immintrin.h"


#define RED 0.299
#define GREEN 0.587
#define BLUE 0.114


//DILOSI SINARTISEO
int convolutionAVX(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp, char * new_name, int filterSize, float multFactor);


int max(int num1, int num2){
    if(num1>num2)
        return num1;
    else
        return num2;
}

int min(int num1, int num2){
    if(num1<num2)
        return num1;
    else
        return num2;
}




void pixelProcessor(pixel **PIXEL, pixel **PIXEL_OUT, int i_WIDTH, int i_HEIGHT, int filterSize, kernel *kernel1, float factor){
    int i;
    float ** filter = kernel1->kernel;
    PIXEL_OUT[i_WIDTH][i_HEIGHT].R = PIXEL[i_WIDTH][i_HEIGHT].R;
    PIXEL_OUT[i_WIDTH][i_HEIGHT].G = PIXEL[i_WIDTH][i_HEIGHT].G;
    PIXEL_OUT[i_WIDTH][i_HEIGHT].B = PIXEL[i_WIDTH][i_HEIGHT].B;

    /**
    Οι πιο κατω ειναι μεταβλητες οι οποιες θα περιεχουν τα αθροισματα των RGB που πολ/στηκαν με τον factor.
    Δεν χρησιμοποιουνται όλα σε σε ολες τις περιπτωσεις. Χρησιμοποουνται μονο οσα χρειαζονται για το καθε φιλτρο.
    */
    __m256 sum1=_mm256_set1_ps(0);
    __m256 sum2=_mm256_set1_ps(0);
    __m256 sum3=_mm256_set1_ps(0);
    __m256 sum4=_mm256_set1_ps(0);
    __m256 sum5=_mm256_set1_ps(0);
    __m256 sum6=_mm256_set1_ps(0);
    __m256 sum7=_mm256_set1_ps(0);
    __m256 sum8=_mm256_set1_ps(0);
    __m256 sum9=_mm256_set1_ps(0);

    /**
	Αρχικοποιηση ενος διανυσματος με τον factor τον οποιο θα πολ/ζουμε καθε pixel.
	*/
    __m256 vector_float1=_mm256_set1_ps(factor);
    __m256 vector_float2=_mm256_set1_ps(factor);
    __m256 vector_float3=_mm256_set1_ps(factor);
    __m256 vector_float4=_mm256_set1_ps(factor);
    __m256 vector_float5=_mm256_set1_ps(factor);
    __m256 vector_float6=_mm256_set1_ps(factor);
    __m256 vector_float7=_mm256_set1_ps(factor);
    __m256 vector_float8=_mm256_set1_ps(factor);
    __m256 vector_float9=_mm256_set1_ps(factor);
    __m256 vector_factor=_mm256_set1_ps(factor);



    float red=0,green=0,blue=0;

    if(filterSize==1){
        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            int start = i - (i_WIDTH - filterSize);
            /*Οταν το φιλτρο ειναι μικρότερο ή ισο απο 4 θα δουλέυουμε το πολύ σε 9(4+4+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη 10 pixels*/
            __m256i int_vector=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);

            /**
            Στις 4 πιο κάτω εντολές διαχωρίζουμε τον  __m256i int_vector σε 4 μεταβλητές τύπου _m128i
            Ο λογός που το κάνω αυτό είναι για να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            Αρα μετατρεπουμε καθε 8αδα 8bit αριθμων σε ακεραιους.
            */
            __m128i part1= {int_vector[0],0};
            __m128i part2= {int_vector[1],0};


            /**
            Χρησιμοποοιώντας την εντολή _mm256_cvtepu8_epi32 μετατρέπουμε τους 8bitous ακεραιους σε ακεραιους των 32 bits.
            Αξίζει να σημειωσω οτι η εντολή αυτή μετατρέπει unsigned 8 bits ακεραιους σε 32 bitous.
            Μετα την εκτέλεση των πιο κάτω εντολών στο x1 βρίσκονται οι πρωτοι 8 αριθμοι , στο χ2 οι επομενοι 8 κοκ. (π.χ x1={x1,x2,x3,x4,x5,x6,x7,x8})
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);


            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);



            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */
            __m128 temp = _mm_load_ps(&filter[start][0] );
            __m256 filter_vector ={temp[0],temp[1],temp[2],temp[3]};

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);

            /**
            Ενα σημαντικο σημειο είναι οι 4 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);
        }
        red=sum1[0]+sum1[3]+sum1[6];
        green=sum1[1]+sum1[4]+sum1[7];
        blue=sum1[2]+sum1[5]+sum2[0];
    } else if (filterSize==2){

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            int start = i - (i_WIDTH - filterSize);
            /*Οταν το φιλτρο ειναι μικρότερο ή ισο απο 4 θα δουλέυουμε το πολύ σε 9(4+4+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη 10 pixels*/
            __m256i int_vector=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);

            /**
            Στις 4 πιο κάτω εντολές διαχωρίζουμε τον  __m256i int_vector σε 4 μεταβλητές τύπου _m128i
            Ο λογός που το κάνω αυτό είναι γι α να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            Αρα μετατρεπουμε καθε 8αδα 8bit αριθμων σε ακεραιους.
            */
            __m128i part1= {int_vector[0],0};
            __m128i part2= {int_vector[1],0};
            /**
            Χρησιμοποοιώντας την εντολή _mm256_cvtepu8_epi32 μετατρέπουμε τους 8bitous ακεραιους σε ακεραιους των 32 bits.
            Αξίζει να σημειωσω οτι η εντολή αυτή μετατρέπει unsigned 8 bits ακεραιους σε 32 bitous.
            Μετα την εκτέλεση των πιο κάτω εντολών στο x1 βρίσκονται οι πρωτοι 8 αριθμοι , στο χ2 οι επομενοι 8 κοκ. (π.χ x1={x1,x2,x3,x4,x5,x6,x7,x8})
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);

            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);

            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            __m256 filter_vector = _mm256_load_ps(&filter[start][0]  );

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],0};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);

            /**
            Ενα σημαντικο σημειο είναι οι 4 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);
        }
        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6];
    } else if (filterSize==3){
        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            int start = i - (i_WIDTH - filterSize);
            /*Οταν το φιλτρο ειναι μικρότερο ή ισο απο 4 θα δουλέυουμε το πολύ σε 9(4+4+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη 10 pixels*/
            __m256i int_vector=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);

            /**
            Στις 4 πιο κάτω εντολές διαχωρίζουμε τον  __m256i int_vector σε 4 μεταβλητές τύπου _m128i
            Ο λογός που το κάνω αυτό είναι γι α να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            Αρα μετατρεπουμε καθε 8αδα 8bit αριθμων σε ακεραιους.
            */
            __m128i part1= {int_vector[0],0};
            __m128i part2= {int_vector[1],0};
            __m128i part3= {int_vector[2],0};

            /**
            Χρησιμοποοιώντας την εντολή _mm256_cvtepu8_epi32 μετατρέπουμε τους 8bitous ακεραιους σε ακεραιους των 32 bits.
            Αξίζει να σημειωσω οτι η εντολή αυτή μετατρέπει unsigned 8 bits ακεραιους σε 32 bitous.
            Μετα την εκτέλεση των πιο κάτω εντολών στο x1 βρίσκονται οι πρωτοι 8 αριθμοι , στο χ2 οι επομενοι 8 κοκ. (π.χ x1={x1,x2,x3,x4,x5,x6,x7,x8})
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);

            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);


            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            __m256 filter_vector = _mm256_load_ps(&filter[start][0]  );

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],0,0,0};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);


            /**
            Ενα σημαντικο σημειο είναι οι 4 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
        }
        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4];
    }
    else if(filterSize==4)
    {

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            int start = i - (i_WIDTH - filterSize);
            /*Οταν το φιλτρο ειναι μικρότερο ή ισο απο 4 θα δουλέυουμε το πολύ σε 9(4+4+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη 10 pixels*/
            __m256i int_vector=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);

            /**
            Στις 4 πιο κάτω εντολές διαχωρίζουμε τον  __m256i int_vector σε 4 μεταβλητές τύπου _m128i
            Ο λογός που το κάνω αυτό είναι γι α να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            Αρα μετατρεπουμε καθε 8αδα 8bit αριθμων σε ακεραιους.
            */
            __m128i part1= {int_vector[0],0};
            __m128i part2= {int_vector[1],0};
            __m128i part3= {int_vector[2],0};
            __m128i part4= {int_vector[3],0};
            /**
            Χρησιμοποοιώντας την εντολή _mm256_cvtepu8_epi32 μετατρέπουμε τους 8bitous ακεραιους σε ακεραιους των 32 bits.
            Αξίζει να σημειωσω οτι η εντολή αυτή μετατρέπει unsigned 8 bits ακεραιους σε 32 bitous.
            Μετα την εκτέλεση των πιο κάτω εντολών στο x1 βρίσκονται οι πρωτοι 8 αριθμοι , στο χ2 οι επομενοι 8 κοκ. (π.χ x1={x1,x2,x3,x4,x5,x6,x7,x8})
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            __m256 filter_vector = _mm256_load_ps(&filter[start][0]  );

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter[start][8],filter[start][8],filter[start][8], 0,0,0,0};




            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);


            /**
            Ενα σημαντικο σημειο είναι οι 4 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);
        }
        /**
        Εδω κανουμε accumulate όλα τα αθροισματα που πήραμε πιο πάνω. Eπομενως αναλογος με το φιλτρο κάνουμε accumulate και τα αντιστοιχα pixels.
        Αρα αν το φιλτρο ηταν 3 τοτε δουλευαμε σε ενα παραθυρο με 3*3 pixels.
         (RGB RGB RGB) * Factor
         (RGB RGB RGB) * Factor
        +(RGB RGB RGB) * Factor
        -----
        3R3G3B 3R3G3B 3R3G3B

        Τα αντιστοιχα αθροισματα βρίσκονται και στις αναλογες θεσεις των sums.
        Αντιστοιχη λογικη εφαρμοστηκε και για πιο μεγαλα φιλτρα οπου το μονο που επρεπε να κανουμε ηταν προσθεσουμε τις αντιστοιχες τιμες των αθροισματων.
        */
        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2];
    }else if(filterSize==5)
    {
        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {


            /*Οταν το φιλτρο ειναι 5 τοτε το παραθυρο το convolution  θα εφαρμοζεται σε ακριβς 11(5+5+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10 αφου το παραθυρο μας ειναι μεγαλυτερο του 10(δλδ 11).
            */
            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);



            /**
            Στις 5 πιο κάτω εντολές διαχωρίζουμε τον  __m256i first_10Pixels σε 4 μεταβλητές τύπου _m128i
            και επίσης περνουμε τα πρωτα 64 bits του __m256i second_10Pixels.
            Ο λογός που το κάνω αυτό είναι για να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            */
            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};


            /**
            Χρησιμοποοιώντας την εντολή _mm256_cvtepu8_epi32 μετατρέπουμε τους 8bitous ακεραιους σε ακεραιους των 32 bits.
            Αξίζει να σημειωσω οτι η εντολή αυτή μετατρέπει unsigned 8 bits ακεραιους σε 32 bitous.
            Μετα την εκτέλεση των πιο κάτω εντολών στο x1 βρίσκονται οι πρωτοι 8 αριθμοι , στο χ2 οι επομενοι 8 κοκ. (π.χ x1={x1,x2,x3,x4,x5,x6,x7,x8})
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);


            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);


            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]  );
            __m256 filter_vector2 = _mm256_load_ps(&filter[start][8]  );

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};

            __m256 vector_float4 = {filter_vector2[0],filter_vector2[0],filter_vector2[0],filter_vector2[1],filter_vector2[1],filter_vector2[1],filter_vector2[2],filter_vector2[2]};
            __m256 vector_float5 = {filter_vector2[2],filter_vector2[3],filter_vector2[3], filter_vector2[3],filter_vector2[4],filter_vector2[4],filter_vector2[4],filter_vector2[5]};

            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);
            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);
            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);


            /**
            Ενα σημαντικο σημειο είναι οι 5 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);
            sum5=_mm256_floor_ps(sum5);
        }

        /**
        Πιο κατω προσθετουμε τα pixels που αντιστοιχουν στα 11 του παραθυρου
        */
        float red=0,green=0,blue=0;
        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2];

        PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);

        return;
    } else if(filterSize == 6){
        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            /*
            Οταν το φιλτρο ειναι 6 τοτε το παραθυρο το convolution  θα εφαρμοζεται σε ακριβως 13(6+6+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10 αφου το παραθυρο μας ειναι μεγαλυτερο του 10(δλδ 13 ).
            ή

            Οταν το φιλτρο ειναι 7 τοτε το παραθυρο το convolution  θα εφαρμοζεται σε ακριβως 15(7+7+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10 αφου το παραθυρο μας ειναι μεγαλυτερο του 10(δλδ 15 ).
            */
            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);


            /**
            Στις 6 πιο κάτω εντολές διαχωρίζουμε τον  __m256i first_10Pixels σε 4 μεταβλητές τύπου _m128i
            και επίσης περνουμε τα πρωτα 128 bits του __m256i second_10Pixels.
            Ο λογός που το κάνω αυτό είναι για να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            */
            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};

            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου του πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);

            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);

            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]);
            __m256 filter_vector1 = _mm256_load_ps(&filter[start][8]);

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter_vector1[0],filter_vector1[0],filter_vector1[0], filter_vector1[1], filter_vector1[1],filter_vector1[1],filter_vector1[2],filter_vector1[2]};
            __m256 vector_float5 = {filter_vector1[2],filter_vector1[3],filter_vector1[3], filter_vector1[3],filter_vector1[4],filter_vector1[4],filter_vector1[4],filter_vector1[5]};
            __m256 vector_float6 = {filter_vector1[5],filter_vector1[5],filter_vector1[6], filter_vector1[6],filter_vector1[6],filter_vector1[7],filter_vector1[7],filter_vector1[7]};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);


            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);

            /**
            Ενα σημαντικο σημειο είναι οι 6 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5 ή αλλιως cast σε int) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);

            sum5=_mm256_floor_ps(sum5);
        }
        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0];
    }
    else if(filterSize==7){

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            /*
            Οταν το φιλτρο ειναι 6 τοτε το παραθυρο το convolution  θα εφαρμοζεται σε ακριβως 13(6+6+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10 αφου το παραθυρο μας ειναι μεγαλυτερο του 10(δλδ 13 ).
            ή

            Οταν το φιλτρο ειναι 7 τοτε το παραθυρο το convolution  θα εφαρμοζεται σε ακριβως 15(7+7+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10 αφου το παραθυρο μας ειναι μεγαλυτερο του 10(δλδ 15 ).
            */
            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);


            /**
            Στις 6 πιο κάτω εντολές διαχωρίζουμε τον  __m256i first_10Pixels σε 4 μεταβλητές τύπου _m128i
            και επίσης περνουμε τα πρωτα 128 bits του __m256i second_10Pixels.
            Ο λογός που το κάνω αυτό είναι για να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            */
            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};
            __m128i part6={second_10Pixels[1],0};

            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου του πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);
            __m256i x6=  _mm256_cvtepu8_epi32 (part6);

            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);
            __m256 x6_float=_mm256_cvtepi32_ps (x6);


            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]);
            __m256 filter_vector1 = _mm256_load_ps(&filter[start][8]);

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter_vector1[0],filter_vector1[0],filter_vector1[0], filter_vector1[1],filter_vector1[1],filter_vector1[1],filter_vector1[2],filter_vector1[2]};
            __m256 vector_float5 = {filter_vector1[2],filter_vector1[3],filter_vector1[3], filter_vector1[3],filter_vector1[4],filter_vector1[4],filter_vector1[4],filter_vector1[5]};
            __m256 vector_float6 = {filter_vector1[5],filter_vector1[5],filter_vector1[6],filter_vector1[6],filter_vector1[6],filter_vector1[7],filter_vector1[7],filter_vector1[7]};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_float6,sum6);


            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_factor,sum6);

            /**
            Ενα σημαντικο σημειο είναι οι 6 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5 ή αλλιως cast σε int) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);

            sum5=_mm256_floor_ps(sum5);
            sum6=_mm256_floor_ps(sum6);



        }

        /**
        Πιο κατω προσθετουμε τα pixels που αντιστοιχουν στα 13 ή 15 του παραθυρου
        */
        if(filterSize==6){
            red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6];
            green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7];
            blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0];
        } else {
            red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6]+sum6[1]+sum6[4];
            green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7]+sum6[2]+sum6[5];
            blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0]+sum6[3]+sum6[6];
        }



        PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);
        return;

    }else if(filterSize==8){//* Δεν υπαρχει περιγραφη για το φιλτρο 8 αφου ειναι ακριβως οι ιδιες πραξεις με τα φιλτρα 6,7 απλα προσθετει τα extra pixels.  /

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {




            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);

            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};
            __m128i part6={second_10Pixels[1],0};
            __m128i part7={second_10Pixels[2],0};

            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);
            __m256i x6=  _mm256_cvtepu8_epi32 (part6);
            __m256i x7=  _mm256_cvtepu8_epi32 (part7);

            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);
            __m256 x6_float=_mm256_cvtepi32_ps (x6);
            __m256 x7_float=_mm256_cvtepi32_ps (x7);

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]);
            __m256 filter_vector1 = _mm256_load_ps(&filter[start][8]);
            __m256 filter_vector2 = _mm256_load_ps(&filter[start][16]);

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter_vector1[0],filter_vector1[0],filter_vector1[0], filter_vector1[1],filter_vector1[1],filter_vector1[1],filter_vector1[2],filter_vector1[2]};
            __m256 vector_float5 = {filter_vector1[2],filter_vector1[3],filter_vector1[3], filter_vector1[3],filter_vector1[4],filter_vector1[4],filter_vector1[4],filter_vector1[5]};
            __m256 vector_float6 = {filter_vector1[5],filter_vector1[5],filter_vector1[6], filter_vector1[6],filter_vector1[6],filter_vector1[7],filter_vector1[7],filter_vector1[7]};
            __m256 vector_float7 = {filter_vector2[0],filter_vector2[0],filter_vector2[0],filter_vector2[1],filter_vector2[1],filter_vector2[1],filter_vector2[2],filter_vector2[2]};



            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_float6,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_float7,sum7);


            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_factor,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_factor,sum7);

            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);

            sum5=_mm256_floor_ps(sum5);
            sum6=_mm256_floor_ps(sum6);
            sum7=_mm256_floor_ps(sum7);
        }


        float red=0,green=0,blue=0;

        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6]+sum6[1]+sum6[4]+sum6[7]+sum7[2];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7]+sum6[2]+sum6[5]+sum7[0]+sum7[3];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0]+sum6[3]+sum6[6]+sum7[1]+sum7[4];




        PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);
        return;

    }else if(filterSize==9 ){//* Δεν υπαρχει περιγραφη για το φιλτρο 9 αφου ειναι ακριβως οι ιδιες πραξεις με τα φιλτρα 6,7 απλα προσθετει τα extra pixels.  /

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);


            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};
            __m128i part6={second_10Pixels[1],0};
            __m128i part7={second_10Pixels[2],0};
            __m128i part8={second_10Pixels[3],0};

            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);
            __m256i x6=  _mm256_cvtepu8_epi32 (part6);
            __m256i x7=  _mm256_cvtepu8_epi32 (part7);
            __m256i x8=  _mm256_cvtepu8_epi32 (part8);

            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);
            __m256 x6_float=_mm256_cvtepi32_ps (x6);
            __m256 x7_float=_mm256_cvtepi32_ps (x7);
            __m256 x8_float=_mm256_cvtepi32_ps (x8);

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]);
            __m256 filter_vector1 = _mm256_load_ps(&filter[start][8]);
            __m256 filter_vector2 = _mm256_load_ps(&filter[start][16]);

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter_vector1[0],filter_vector1[0],filter_vector1[0], filter_vector1[1],filter_vector1[1],filter_vector1[1],filter_vector1[2],filter_vector1[2]};
            __m256 vector_float5 = {filter_vector1[2],filter_vector1[3],filter_vector1[3], filter_vector1[3],filter_vector1[4],filter_vector1[4],filter_vector1[4],filter_vector1[5]};
            __m256 vector_float6 = {filter_vector1[5],filter_vector1[5],filter_vector1[6], filter_vector1[6],filter_vector1[6],filter_vector1[7],filter_vector1[7],filter_vector1[7]};
            __m256 vector_float7 = {filter_vector2[0],filter_vector2[0],filter_vector2[0],filter_vector2[1],filter_vector2[1],filter_vector2[1],filter_vector2[2],filter_vector2[2]};
            __m256 vector_float8 = {filter_vector2[2],filter_vector2[3],filter_vector2[3], filter_vector2[3],filter_vector2[4],filter_vector2[4],filter_vector2[4],filter_vector2[5]};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_float6,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_float7,sum7);
            sum8=_mm256_fmadd_ps(x8_float,vector_float8,sum8);

            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_factor,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_factor,sum7);
            sum8=_mm256_fmadd_ps(x8_float,vector_factor,sum8);

            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);

            sum5=_mm256_floor_ps(sum5);
            sum6=_mm256_floor_ps(sum6);
            sum7=_mm256_floor_ps(sum7);
            sum8=_mm256_floor_ps(sum8);
        }


        float red=0,green=0,blue=0;

        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6]+sum6[1]+sum6[4]+sum6[7]+sum7[2]+sum7[5]+sum8[0];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7]+sum6[2]+sum6[5]+sum7[0]+sum7[3]+sum7[6]+sum8[1];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0]+sum6[3]+sum6[6]+sum7[1]+sum7[4]+sum7[7]+sum8[2];




        PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);
        return;

    }else if(filterSize==10 ){

        for (i = i_WIDTH - filterSize; i <= i_WIDTH + filterSize; i++)
        {
            /*
            Οταν το φιλτρο ειναι 10 τοτε το παραθυρο του convolution  θα εφαρμοζεται σε ακριβως 21(10+10+1) pixels σε κάθε επανάληψη, εδώ κάνουμε load 256 bits,δηλαδη τα πρωτα 10 pixels
            και ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10. Χρειαζομαστε ακομα 1 load αφου το παραθυρο δουλευει σε 21 pixels και ως τωρα φορτωσαμε 20.
            Ακολουθως ακομα 1 load για να φορτωσουμε αλλα 10.
            */
            __m256i first_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize]);
            __m256i second_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10]);
            __m256i third_10Pixels=_mm256_loadu_si256((__m256i * )&PIXEL[i][i_HEIGHT - filterSize+10+10]);


            /**
            Στις 9 πιο κάτω εντολές διαχωρίζουμε τον  __m256i first_10Pixels σε 4 μεταβλητές τύπου _m128i,
            επίσης διαχωρίζουμε τον  __m256i first_10Pixels σε 4 μεταβλητές τύπου _m128i
            και επίσης περνουμε τα πρωτα 64 bits του __m256i third_10Pixels.
            Ο λογός που το κάνω αυτό είναι για να χρησιμοποιησώ τη μέθοδο _mm256_cvtepu8_epi32  η οποία
            δέχεται μόνο __m128i μεταβλητές. Στο part1 μπαινουν τα πρωτα 64 bits,στο part2 τα επομενα 64 bits κοκ. (π.χ part1={x1,x2,x3,x4,x5,x7,x8,0,0,0,0,0,0,0,0})
            Ο λογός που μπένουν μονο τα πρωτα 64 είναι διοτι η εντολή _mm256_cvtepu8_epi32 επεξεργάζεται μόνο τους πρωτους 8 αριθμους.
            Επομενως με αυτο το τροπο έχουμε και τα 21 pixels με τα οποια θελουμε να δουλεψουμε.
            Στην ακριβεια κανουμε πραξη με 30 pixels αλλα στο τελος πέρνουμε μονο τα 21 που μας ενδιαφερουν.
            */
            __m128i part1= {first_10Pixels[0],0};
            __m128i part2= {first_10Pixels[1],0};
            __m128i part3= {first_10Pixels[2],0};
            __m128i part4= {first_10Pixels[3],0};

            __m128i part5={second_10Pixels[0],0};
            __m128i part6={second_10Pixels[1],0};
            __m128i part7={second_10Pixels[2],0};
            __m128i part8={second_10Pixels[3],0};

            __m128i part9={third_10Pixels[0],0};


            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου του πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */
            __m256i x1=  _mm256_cvtepu8_epi32 (part1);
            __m256i x2=  _mm256_cvtepu8_epi32 (part2);
            __m256i x3=  _mm256_cvtepu8_epi32 (part3);
            __m256i x4=  _mm256_cvtepu8_epi32 (part4);

            __m256i x5=  _mm256_cvtepu8_epi32 (part5);
            __m256i x6=  _mm256_cvtepu8_epi32 (part6);
            __m256i x7=  _mm256_cvtepu8_epi32 (part7);
            __m256i x8=  _mm256_cvtepu8_epi32 (part8);

            __m256i x9=  _mm256_cvtepu8_epi32 (part9);


            /**
            Χρησιμοποοιώντας την εντολή  _mm256_cvtepi32_ps μετατρέπουμε 32bitous ακεραιους σε floats. Ο λόγος που το κάνουμε αυτό είναι διοτι καθε αριθμός πρέπει να
            πολ/στει με ενα factor ο οποίος είναι float.
            */
            __m256 x1_float= _mm256_cvtepi32_ps(x1);
            __m256 x2_float= _mm256_cvtepi32_ps(x2);
            __m256 x3_float= _mm256_cvtepi32_ps(x3);
            __m256 x4_float= _mm256_cvtepi32_ps(x4);

            __m256 x5_float=_mm256_cvtepi32_ps (x5);
            __m256 x6_float=_mm256_cvtepi32_ps (x6);
            __m256 x7_float=_mm256_cvtepi32_ps (x7);
            __m256 x8_float=_mm256_cvtepi32_ps (x8);

            __m256 x9_float=_mm256_cvtepi32_ps (x9);



            /**
            Στις πιο κατω εντολες πολ/ζουμε και τους αριθμους που μετατρεψαμε σε floats με τον factor και ακολουθως προσθετουμε το αποτελεσμα αυτου το πολ/σμου στις μεταβλητες sum.
            π.χ : Αν x1_float={R,G,B,R,G,B,R,G} και sum1 ={a1,a2,a3,a4,a5,a6,a7,a8} τοτε μετα την εφαρμογη αυτης της fmadd θα προκύψει
            sum1={a1+R*factor,a2+G*factor,a3+B*factor,a4+R*factor,a5+G*factor,a6+B*factor,a7+R*factor,a8+G*factor}.
            Επισης το a1 αντιστοιχει στο red, το a2 στο green , το a3 στο blue , το a4->red κοκ.
            Επομενως όταν τελειωσουν όλες οι επαναληψεις θα μπορουμε να μαζεψουμε όλα τα αθροισματα των R,G,B αφου θα ξέρουμε που θα βρίσκονται.
            */

            int start = i - (i_WIDTH - filterSize);
            __m256 filter_vector = _mm256_load_ps(&filter[start][0]);
            __m256 filter_vector1 = _mm256_load_ps(&filter[start][8]);
            __m256 filter_vector2 = _mm256_load_ps(&filter[start][16]);

            __m256 vector_float1 = {filter_vector[0],filter_vector[0],filter_vector[0], filter_vector[1],filter_vector[1],filter_vector[1] ,filter_vector[2],filter_vector[2]};
            __m256 vector_float2 = {filter_vector[2],filter_vector[3],filter_vector[3], filter_vector[3],filter_vector[4],filter_vector[4] ,filter_vector[4],filter_vector[5]};
            __m256 vector_float3 = {filter_vector[5],filter_vector[5],filter_vector[6], filter_vector[6],filter_vector[6],filter_vector[7] ,filter_vector[7],filter_vector[7]};
            __m256 vector_float4 = {filter_vector1[0],filter_vector1[0],filter_vector1[0], filter_vector1[1],filter_vector1[1],filter_vector1[1],filter_vector1[2],filter_vector1[2]};
            __m256 vector_float5 = {filter_vector1[2],filter_vector1[3],filter_vector1[3], filter_vector1[3],filter_vector1[4],filter_vector1[4],filter_vector1[4],filter_vector1[5]};
            __m256 vector_float6 = {filter_vector1[5],filter_vector1[5],filter_vector1[6], filter_vector1[6],filter_vector1[6],filter_vector1[7],filter_vector1[7],filter_vector1[7]};
            __m256 vector_float7 = {filter_vector2[0],filter_vector2[0],filter_vector2[0],filter_vector2[1],filter_vector2[1],filter_vector2[1],filter_vector2[2],filter_vector2[2]};
            __m256 vector_float8 = {filter_vector2[2],filter_vector2[3],filter_vector2[3], filter_vector2[3],filter_vector2[4],filter_vector2[4],filter_vector2[4],filter_vector2[5]};
            __m256 vector_float9 = {filter_vector2[5],filter_vector2[5],filter_vector2[6], filter_vector2[6],filter_vector2[6],filter_vector2[7],filter_vector2[7],filter_vector2[7]};


            sum1=_mm256_fmadd_ps(x1_float,vector_float1,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_float2,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_float3,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_float4,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_float5,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_float6,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_float7,sum7);
            sum8=_mm256_fmadd_ps(x8_float,vector_float8,sum8);
            //sum9=_mm256_fmadd_ps(x9_float,vector_float9,sum9);


            sum1=_mm256_fmadd_ps(x1_float,vector_factor,sum1);
            sum2=_mm256_fmadd_ps(x2_float,vector_factor,sum2);
            sum3=_mm256_fmadd_ps(x3_float,vector_factor,sum3);
            sum4=_mm256_fmadd_ps(x4_float,vector_factor,sum4);

            sum5=_mm256_fmadd_ps(x5_float,vector_factor,sum5);
            sum6=_mm256_fmadd_ps(x6_float,vector_factor,sum6);
            sum7=_mm256_fmadd_ps(x7_float,vector_factor,sum7);
            sum8=_mm256_fmadd_ps(x8_float,vector_factor,sum8);
            //sum9=_mm256_fmadd_ps(x9_float,vector_factor,sum9);


            /**
            Ενα σημαντικο σημειο είναι οι 9 πιο κατω εντολες. Στο σεριαλ οταν πολ/ζει τον ακεραι με τον float μετα κανει truncate(π.χ 5.64->5) τον αριθμο που προκυπτει απο τον πολ/σμο.
            Αυτος ειναι και ο λογος που χρησιμοποιησώ την πιο κατω εντολη ετσι ωστε αν sum1={4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8} -> sum1={4,4,4,4,4,4,4,4}
            */
            sum1=_mm256_floor_ps(sum1);
            sum2=_mm256_floor_ps(sum2);
            sum3=_mm256_floor_ps(sum3);
            sum4=_mm256_floor_ps(sum4);

            sum5=_mm256_floor_ps(sum5);
            sum6=_mm256_floor_ps(sum6);
            sum7=_mm256_floor_ps(sum7);
            sum8=_mm256_floor_ps(sum8);

            sum9=_mm256_floor_ps(sum9);
        }

        float red=0,green=0,blue=0;

        red=sum1[0]+sum1[3]+sum1[6]+sum2[1]+sum2[4]+sum2[7]+sum3[2]+sum3[5]+sum4[0]+sum4[3]+sum5[0]+sum5[3]+sum5[6]+sum6[1]+sum6[4]+sum6[7]+sum7[2]+sum7[5]+sum8[0]+sum8[3]+sum9[0];
        green=sum1[1]+sum1[4]+sum1[7]+sum2[2]+sum2[5]+sum3[0]+sum3[3]+sum3[6]+sum4[1]+sum4[4]+sum5[1]+sum5[4]+sum5[7]+sum6[2]+sum6[5]+sum7[0]+sum7[3]+sum7[6]+sum8[1]+sum8[4]+sum9[1];
        blue=sum1[2]+sum1[5]+sum2[0]+sum2[3]+sum2[6]+sum3[1]+sum3[4]+sum3[7]+sum4[2]+sum4[5]+sum5[2]+sum5[5]+sum6[0]+sum6[3]+sum6[6]+sum7[1]+sum7[4]+sum7[7]+sum8[2]+sum8[5]+sum9[2];




        PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
        PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);
        return;
    }
    PIXEL_OUT[i_WIDTH][i_HEIGHT].R =  min( max( (int)(kernel1->factor * red   + kernel1->bias),   0), 255);
    PIXEL_OUT[i_WIDTH][i_HEIGHT].G =  min( max( (int)(kernel1->factor * green + kernel1->bias), 0), 255);
    PIXEL_OUT[i_WIDTH][i_HEIGHT].B =  min( max( (int)(kernel1->factor * blue  + kernel1->bias),  0), 255);

}



kernel * getKernel(int filterSize){
    kernel *retKernel= malloc(sizeof (kernel));
    int size = filterSize*2+1;
    float  **filter= (float **)aligned_alloc(32,size*sizeof (float *));
    for (int k = 0; k <size ; ++k) {
        filter[k] = (float *)aligned_alloc(32,size*sizeof (float ));
    }
    //If filterSize = 1 then return a 3x3 filter for edge detection.
    if (filterSize==1){
        filter[0][0]=-1;filter[0][1]=-1;filter[0][2]=-1;
        filter[1][0]=-1;filter[1][1]=8;filter[1][2]=-1;
        filter[2][0]=-1;filter[2][1]=-1;filter[2][2]=-1;
        retKernel->factor = 1;
        retKernel->bias = 0;
        retKernel->kernel=filter;
    }
        //If filterSize = 2 return a 5x5 Kernel for edge detection.
    else if (filterSize==2){
        filter[0][0]=-1;filter[0][1]=0;filter[0][2]=0;filter[0][3]=0;filter[0][4]=0;
        filter[1][0]=0;filter[1][1]=-2;filter[1][2]=0;filter[1][3]=0;filter[1][4]=0;
        filter[2][0]=0;filter[2][1]=0;filter[2][2]=6;filter[2][3]=0;filter[2][4]=0;
        filter[3][0]=0;filter[3][1]=0;filter[3][2]=0;filter[3][3]=-2;filter[3][4]=0;
        filter[4][0]=0;filter[4][1]=0;filter[4][2]=0;filter[4][3]=0;filter[4][4]=-1;
        retKernel->factor = 1;
        retKernel->bias = 0;
        retKernel->kernel=filter;
    } else {
        for (int i = 0; i <size ; i++) {
            for(int j=0;j<size;j++)
                if(i==j)
                    filter[i][j]=1;
        }
        retKernel->factor = 1.0f/9.0f;
        retKernel->bias = 0;
        retKernel->kernel = filter;
    }
    return retKernel;
}




pixel **createImage(BMP_INFO_HEADER *BIH, int padding){
    pixel **PIXEL;
    //Allocate space to create a 2D array where the output image is going to be stored.
    if ((PIXEL = (pixel **) malloc((BIH->imageHeight) * sizeof(pixel *))) == NULL)
    {
        printf("Can't allocate array");
        exit(-1);
    }

    for (int i = 0; i < BIH->imageHeight; i++){
        if ((PIXEL[i] = (pixel *) malloc((BIH->imageWidth + padding) * sizeof(pixel))) == NULL)
        {
            printf("Can't allocate array");
            exit(-1);
        }
    }
    return PIXEL;
}

void readImage(BMP_INFO_HEADER *BIH, int padding, FILE *fp, pixel **PIXEL){
    for (int row = 0; row < (BIH->imageHeight); row++) {
        for (int col = 0; col < (BIH->imageWidth + padding); col++) {

            if (col >= BIH->imageWidth)
                fread((void *) &(PIXEL[row][col]), sizeof(myByte), 1, fp);
            else
                fread((void *) &(PIXEL[row][col]), sizeof(pixel), 1, fp);
        }
    }
}

// Convolution Function
int convolutionAVX(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp, char * new_name, int filterSize, float multFactor) {

    int padding;
    if (((BIH->imageWidth * 3) % 4) == 0)
        padding = 0;
    else
        padding = 4 - ((3 * BIH->imageWidth) % 4);

    pixel **inputImage = NULL, **outputImage=NULL;


    inputImage = createImage(BIH, padding);
    outputImage = createImage(BIH, padding);
    readImage(BIH, padding, fp, inputImage);
    kernel *kernel1 = getKernel(filterSize);
    fclose(fp);

    // This loop iterates over the rows and columns of a 2D image array, applying a pixel processing function to the pixels within a filter size,
    // and setting the pixel to a new value if it falls outside of the filter area. The loop uses row and col as loop variables to represent the current pixel coordinates,
    // and checks that the pixel is within the filter area before calling the pixel processing function.If the pixel is outside of the filter area, it calculates a new pixel
    // value based on the original pixel's color values and sets the pixel to the new value.
    for (int row = filterSize; row < (BIH->imageHeight) - filterSize; row++) {
        for (int col = filterSize; col < (BIH->imageWidth) - filterSize; col++) {
            if ((row > filterSize) && (row < (BIH->imageHeight) - filterSize) && (col > filterSize) && (col < (BIH->imageWidth) - filterSize)) {
                pixelProcessor(inputImage, outputImage, row, col, filterSize, kernel1, multFactor);
            } else {
                myByte NEW_PIXEL;
                NEW_PIXEL = (int) (inputImage[row][col].R * RED + inputImage[row][col].G * GREEN + inputImage[row][col].B * BLUE);
                outputImage[row][col].R = NEW_PIXEL;
                outputImage[row][col].G = NEW_PIXEL;
                outputImage[row][col].B = NEW_PIXEL;
            }
        }
    }


#ifdef DEBUG
    FILE *outputFile;
    outputFile=fopen(new_name, "wb");

    writeBmpInformation(BFH, BIH, outputFile);
    // Write the image data to the output file, including padding bytes
    for(int i_WIDTH=0; i_WIDTH<(BIH->imageHeight); i_WIDTH++) {
        for(int i_HEIGHT=0; i_HEIGHT<(BIH->imageWidth + padding); i_HEIGHT++) {

            // If the current column is outside the image bounds, write a padding byte
            if(i_HEIGHT>=BIH->imageWidth)
                fwrite((void *)&(outputImage[i_WIDTH][i_HEIGHT]), sizeof(myByte), 1, outputFile);
            // Otherwise, write a pixel from the output image
            else
                fwrite((void *)&(outputImage[i_WIDTH][i_HEIGHT]), sizeof(pixel), 1, outputFile);
        }
    }

    // Close the output file
    fclose(outputFile);
#else
#endif
    for (int i = 0; i < BIH->imageHeight; i++)
        free(inputImage[i]);

    free(inputImage);
    return (0);
}


int main(int argc, char *argv[]) {

    // Declare variables for BMP file headers, input file pointer, and output file name.
    BMP_FILE_HEADER bfh;
    BMP_INFO_HEADER bih;
    FILE *fp;
    char convolvedImageName[128];

    // Define constants for factor step and end value.
    const float FACTOR_STEP = 0.005;
    const float FACTOR_END = 0.1;

    // Check if the command line arguments are valid.
    if (argc != 3){
        printf("Wrong Command Line. Format:./a.out FilterSize BMPimageName.bmp\n");
        return 0;
    }


#ifdef DEBUG
    printf("Convolution process started...\n");
#endif

    // Start timing the program.
    startTime(0);

    // First Argument is for the Filter Size
    int filterSize = atoi(argv[1]);

    for (float factor = 0.00; factor <= FACTOR_END; factor += FACTOR_STEP) {

        // Open the input BMP file
        fp = fopen(argv[2], "rb");

        // If the BMP file header and information header cannot be loaded, terminate the program.
        if (load_bmp_headers(&bfh, &bih, fp) == 1) continue;

        // Construct the output BMP file name using the filter size, factor, and input BMP file name.
        sprintf(convolvedImageName, "output//conv_%dx%d_%3.2f_%s", (2 * filterSize + 1), (2 * filterSize + 1), factor, argv[2]);

#ifdef DEBUG
        printf("Conversion of %s: Factor:%f, Filter Size %dx%d:\n",argv[1],factor,filterSize,filterSize);
        print_header_info(&bfh, &bih);
#else
#endif

        // Call the CUDA convolution function to convolve the input BMP file with the filter.
        convolutionAVX(&bfh, &bih, fp, convolvedImageName, filterSize, factor);

    }
    stopTime(0);
    elapsedTime(0);
    return 0;
}
#endif