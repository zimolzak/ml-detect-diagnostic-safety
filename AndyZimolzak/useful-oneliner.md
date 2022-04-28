    git ls-files | grep -v '/' | perl -ne 'chomp; $x=$_; $x =~ s/.*\.//; print "$x $_\n"' | sort

This lists files tracked by git, excludes those in directories, prints
the file extension first, and sorts. Useful for deciding what to move
out of the git root directory when tidying up the repo.
