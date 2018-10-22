---
title: Static analysis
category: cpp
layout: main
---

The goal of this exercise is to learn how to use `clang-tidy` to verify that
your code satisfies certain rules. The checks are applied statically, i.e.
without running the code.

Consider again [hello.cc]({{site.exercises_repo}}/hands-on/basic/hello.cc), in
its formatted version.

    [studentNM@esc-XY ~]$ cd esc18/hands-on/basic
    [studentNM@esc-XY basic]$ cat hello.cc
    #include <iostream>
    int main(int argc, char* argv[]) {
      std::cout << "Hello, ";
      if (argc > 1)
        std::cout << argv[1];
    ...

Note how the statements in the `if` and `else` branches are not included in a
block, i.e. they are not surrounded by braces, which is something that we want
to have.

    [studentNM@esc-XY basic]$ clang-tidy -checks=readability-braces-around-statements hello.cc
    Error while trying to load a compilation database:
    ...
    /export/students-home/student40/esc18/hands-on/basic/hello.cc:4:16: warning: statement should be inside braces [readability-braces-around-statements]
      if (argc > 1)
                   ^
                    {
    ...

Let's ignore for a moment the compilation database error. Let's ask `clang-tidy`
to also add the braces (`-fix`) and, since we have already a `.clang-format`, to also
reformat the resulting code (`-format-style=file`).

    [studentNM@esc-XY basic]$ clang-tidy -checks=readability-braces-around-statements -fix -format-style=file hello.cc
    ...
    [studentNM@esc-XY basic]$ cat hello.cc
    #include <iostream>
    int main(int argc, char*argv[]){std::cout<<"Hello, ";
      if (argc > 1) {
        std::cout << argv[1];
      } else {
        std::cout << "world";
      }
      std::cout << "!\n";
    }

The result is not always perfect. Better rerun `clang-format` manuall

    [studentNM@esc-XY basic]$ clang-format -i hello.cc
    [studentNM@esc-XY basic]$ cat hello.cc
    #include <iostream>
    int main(int argc, char* argv[]) {
      std::cout << "Hello, ";
      if (argc > 1) {
        std::cout << argv[1];
      } else {
        std::cout << "world";
      }
      std::cout << "!\n";
    }

Tens of checks are available. To list the enabled checks:

    [studentNM@esc-XY basic]$ clang-tidy -list-checks

To list all the available checks:

    [studentNM@esc-XY basic]$ clang-tidy -checks=* -list-checks

To dump a configuration file including all the checks but the mpi ones:

    [studentNM@esc-XY basic]$ clang-tidy -checks=*,-mpi* -dump-config

Coming back to the "compilation database", the easiest way to create one is
through `cmake`. The compilation database is a file called
`compile_commands.json`.

    [studentNM@esc-XY basic]$ mkdir build && cd build
    [studentNM@esc-XY build]$ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
    [studentNM@esc-XY build]$ cd ..
    [studentNM@esc-XY basic]$ git checkout -- hello.cc
    [studentNM@esc-XY basic]$ clang-tidy -checks=readability-braces-around-statements -p build hello.c
