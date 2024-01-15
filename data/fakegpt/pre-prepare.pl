#!/usr/bin/perl

use strict;
use warnings;
use Text::CSV;
use File::Basename;

if (@ARGV < 1) {
    die "usage: pre-prepare.pl <input>\n";
}

my $input = $ARGV[0];
my $basename = basename($input);
my $dirname = dirname($input);
my $output = $dirname . '/_' . $basename;

my $csv = Text::CSV->new({
    binary => 1, # Permitir caracteres binários
    auto_diag => 1, # Relatar erros automaticamente
    eol => "\n", # Definir o fim da linha
    quote_char => undef,  # Desabilita o tratamento especial de aspas duplas
    escape_char => undef  # Desabilita o escape de caracteres
});

open(my $rinput, "<:encoding(utf8)", $input) or die "unable to open input: $!";
open(my $routput, ">:encoding(utf8)", $output) or die "unable to open output: $!";

while (my $row = $csv->getline($rinput)) {
    
    if (@$row >= 3) {
        my $col = join(",", @$row[3..$#$row]);
        $col =~ s/"//g;
        $row->[3] = "\"" . $col . "\"";
        # $row->[3] = $col;
        @$row = @$row[0..3];

        # Escreve a linha modificada no arquivo de saída
        $csv->print($routput, $row);
        # print $routput "\n";
    }
}

close $rinput;
close $routput;

print "the file $output was created with the modifications.\n";
