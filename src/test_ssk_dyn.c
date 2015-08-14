#include <stdio.h>
#include "ssk_dyn.h"

int main(int argc, char *argv[]) {

    char *s1[] = {"neel","neel","neel"};
    char *s2[] = {"neel","neel","neel"};

    printf("Word sequence kernel value %f\n",ssk_dyn(s1,3,s2,3,2,0.5));
    return 0;
}
