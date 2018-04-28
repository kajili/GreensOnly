%CODE FOUND ONLINE, ATTEMPTS TO FIND STRAIGHT LINES, STILL TRYING TO MAKE
%IT WORK


clear all
close all
clc
image=imread('test1.png');

%//////////////////////////////////////////

image_g=rgb2gray(image);
figure(3)
imshow(image_g)
drawnow
dummy(1:size(image_g,1),1)=0;
a=[dummy image_g(:,1:end-1)];

c=double(a)-double(image_g);
c=c>20;
figure(4)
imshow(c)
 drawnow
c=bwareaopen(c,30);
figure(5)
imshow(c)
drawnow
[m1,n1]=size(c);

for i=150:m1
    [lab,num]=bwlabel(c(i,:));
    a1(i,:)=[i num];
end
a1=sortrows(a1,2);
r_all=a1(end,1);

%/////////////////////////////////////////////

[lab,num]=bwlabel(c);
for i=1:num
    [r,c]=find(lab==i);
    [r1,c1]=find(r==r_all);
    if(length(r1)>0)
        a=length(r);
        mat(i,:)=[i a];
    end
end
mat=sortrows(mat,2);
mat=flipud(mat);

%///////////////////////////////////////////////

img_1(1:m1,1:n1)=0;
img_1=logical(img_1);

for i=1:length(mat)
                [r,c]=find(lab==mat(i,1));
                at=[r c];
                [r1,c1]=find(r==min(r));
                num_1=[1 at(r1(1),2)];

                  [r1,c1]=find(r==max(r));
                  num_2=[m1 at(r1(1),2)];

                  %//////////////////////////////////////////////

                      startp=num_1;
                      endp=num_2;

                      pts=2000;
                       m=atan((endp(2)-startp(2))/(endp(1)-startp(1))); %gradient 
  %                     
  %                     x=m1/2+m1*cos(m*3.14/180)
  %                     y=n1/2+m1*sin(m*3.14/180)

                      if (m==Inf || m==-Inf) %vertical line
                          xx(1:pts)=startp(1);
                          yy(1:pts)=linspace(startp(2),endp(2),pts);
                      elseif m==0 %horizontal line
                          xx(1:pts)=linspace(startp(1),endp(1),pts);
                          yy(1:pts)=startp(2);
                      else %if (endp(1)-startp(1))~=0
                          xx=linspace(startp(1),endp(1),pts);
                          yy=m*(xx-startp(1))+startp(2);
                      end

                      xx=round(xx);
                      yy=round(yy);

                      for j=1:length(xx)
                          if(xx(j)>0 && yy(j)>0 && xx(j)<m1 && yy(j)<n1)
                              img_1(xx(j),yy(j))=1;  
                          end
                      end

figure(7)
imshow(img_1)
drawnow
%/////////////////////////////////////////////////////////////////
end
%https://www.mathworks.com/matlabcentral/answers/260393-how-can-i-get-clear-straight-line-edge-detection-i-have-used-following-image-and-i-have-following