#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void print_features(double *features, int order)
{
  int len = (order+1)*(order+1);
  int i;
  for(i = 0; i<len;i++)
    printf("%le\n",features[i]);
}

// returns an array of size (order+1)^2
double* get_features(double x, double xdot, int order)
{
  double *features = malloc((order+1)*(order+1)*sizeof(double));
  int i,j,index;
  
  /*  if(features==NULL)
    scream_and_die("unable to allocate memory in get_features\n");
  */
  for(i=0;i<=order;i++)
    for(j=0;j<=order;j++)
      {
	index = (order+1)*i + j;
	features[index] = cos(M_PI * ((i * x) + (j * xdot)));
      }

  return features;
}

int main()
{
  print_features(get_features(0.5,0.25,3),3);

  return 0;
}
