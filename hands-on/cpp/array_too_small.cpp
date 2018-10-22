#include <cstdio>
#include <cstring>

int g(char* buf)
{
  return sprintf(buf, "This message will not quite fit into the buf.");
}

int main()
{
  char buf[10];
  int len = g(buf);
  printf("buf = '%s', len = %d", buf, len);
}
