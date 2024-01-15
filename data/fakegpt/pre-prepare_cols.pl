#!/usr/bin/perl

use strict;
use warnings;
use Text::CSV;
use File::Basename;

if (@ARGV < 1) {
    die "usage: pre-prepare_cols.pl <input>\n";
}

my $input = $ARGV[0];
my $basename = basename($input);
my $dirname = dirname($input);
my $output = $dirname . '/_' . $basename;

my $csv = Text::CSV->new({
    binary => 1, # Permitir caracteres binários
    auto_diag => 1, # Relatar erros automaticamente
    eol => "\n", # Definir o fim da linha
#    quote_char => undef,  # Desabilita o tratamento especial de aspas duplas
#    escape_char => undef  # Desabilita o escape de caracteres
});

open my $rinput, "<:encoding(utf8)", $input or die "unable to open input: $!";
open(my $routput, ">:encoding(utf8)", $output) or die "unable to open output: $!";

while (my $row = $csv->getline($rinput)) {
    # $row->[-1] = "\"" . $row->[-1] . "\"";
    my @selected_columns = ($row->[0], $row->[1], $row->[2], $row->[-1]);
    
    $csv->print($routput, \@selected_columns);
}

close $rinput;
close $routput;

print "the file $output was created with the modifications.\n";
