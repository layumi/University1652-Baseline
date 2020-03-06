run('vlfeat-0.9.21/toolbox/vl_setup')
Ia = imresize( imread('1.png'), [480,640]) ;
Ib = imresize( imread('2.png') , [480,640]);
Ic = imresize( imread('3.png') , [480,640]);
%Ia = imresize( vl_impattern('roofs1'), [480,640]) ;
%Ib = imresize( vl_impattern('roofs2') , [480,640]);
%Ic = imresize( vl_impattern('roofs2') , [480,640]);
s = size(Ia);
h = s(1);
w = s(2);

imagesc( 0, 0, Ia);
hold on;
offset1_x = 640 ;
offset1_y = 0 ;
imagesc( offset1_x, offset1_y, Ib);
hold on;
offset2_x = 320 ;
offset2_y = 480 ;
imagesc( offset2_x, offset2_y, Ic);

draw_sift_line(Ia,Ib, 0, 0, offset1_x, offset1_y );
draw_sift_line(Ia,Ic, 0, 0, offset2_x, offset2_y );
draw_sift_line(Ib,Ic, offset1_x, offset1_y, offset2_x, offset2_y );
