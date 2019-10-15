let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/PythonProjects/attosecond5/v3
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +713 ./tf_functions.py
badd +112 xuv_spectrum/spectrum.py
badd +123 ~/.vimrc
badd +1 makecsv.py
badd +1 ./
badd +29 xuv_spectrum/sample4/spectrum4_electron_gen.csv
badd +1587 term://.//40529:/bin/bash
badd +25 fields_window_fft.py
badd +26 phase_parameters/params.py
badd +76 measured_trace/get_trace.py
badd +301 ~/PythonProjects/attosecond5/v3/measured_trace/sample4/energy.csv
badd +301 ~/PythonProjects/attosecond5/v3/measured_trace/sample4/energy_gen.csv
badd +71 term://.//60898:/bin/bash
badd +0 term://.//61891:/bin/bash
badd +0 term://.//64006:/bin/bash
argglobal
%argdel
$argadd .
set stal=2
edit ./tf_functions.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 2resize ' . ((&columns * 52 + 105) / 211)
exe '3resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 3resize ' . ((&columns * 52 + 105) / 211)
exe '4resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 4resize ' . ((&columns * 52 + 105) / 211)
exe '5resize ' . ((&lines * 13 + 28) / 56)
exe 'vert 5resize ' . ((&columns * 52 + 105) / 211)
exe '6resize ' . ((&lines * 12 + 28) / 56)
exe 'vert 6resize ' . ((&columns * 52 + 105) / 211)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
277
normal! zo
479
normal! zo
654
normal! zo
let s:l = 701 - ((34 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
701
normal! 0
wincmd w
argglobal
if bufexists("./tf_functions.py") | buffer ./tf_functions.py | else | edit ./tf_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
277
normal! zo
479
normal! zo
654
normal! zo
let s:l = 186 - ((13 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
186
normal! 017|
wincmd w
argglobal
if bufexists("makecsv.py") | buffer makecsv.py | else | edit makecsv.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 19 - ((11 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
19
normal! 030|
wincmd w
argglobal
if bufexists("./tf_functions.py") | buffer ./tf_functions.py | else | edit ./tf_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
277
normal! zo
479
normal! zo
654
normal! zo
let s:l = 292 - ((9 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
292
normal! 05|
wincmd w
argglobal
if bufexists("term://.//61891:/bin/bash") | buffer term://.//61891:/bin/bash | else | edit term://.//61891:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 905 - ((12 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
905
normal! 0
wincmd w
argglobal
if bufexists("term://.//60898:/bin/bash") | buffer term://.//60898:/bin/bash | else | edit term://.//60898:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 248 - ((11 * winheight(0) + 6) / 12)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
248
normal! 0
wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 2resize ' . ((&columns * 52 + 105) / 211)
exe '3resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 3resize ' . ((&columns * 52 + 105) / 211)
exe '4resize ' . ((&lines * 26 + 28) / 56)
exe 'vert 4resize ' . ((&columns * 52 + 105) / 211)
exe '5resize ' . ((&lines * 13 + 28) / 56)
exe 'vert 5resize ' . ((&columns * 52 + 105) / 211)
exe '6resize ' . ((&lines * 12 + 28) / 56)
exe 'vert 6resize ' . ((&columns * 52 + 105) / 211)
tabedit measured_trace/get_trace.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 16 + 28) / 56)
exe '2resize ' . ((&lines * 36 + 28) / 56)
argglobal
if bufexists("term://.//64006:/bin/bash") | buffer term://.//64006:/bin/bash | else | edit term://.//64006:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 13 - ((9 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 049|
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 74 - ((13 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
74
normal! 0
wincmd w
exe '1resize ' . ((&lines * 16 + 28) / 56)
exe '2resize ' . ((&lines * 36 + 28) / 56)
tabnext 1
set stal=1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFc
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
