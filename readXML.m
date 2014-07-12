% you need these jar files from e.g.
% https://www.apache.org/dist/xerces/j/Xerces-J-bin.2.11.0.tar.gz
%
% you need use javaaddpath to include these files
% javaaddpath('/scratch/ulg/gher/abarth/Downloads/xerces-2_11_0/xercesImpl.jar');
% javaaddpath('/scratch/ulg/gher/abarth/Downloads/xerces-2_11_0/xml-apis.jar');


filename = 'sample.xml';

% these 3 lines are equivalent to xDoc = xmlread(filename)
parser = javaObject('org.apache.xerces.parsers.DOMParser');
parser.parse(filename); % it seems that cd in octave are taken into account
xDoc = parser.getDocument;

% get first data element
elem = xDoc.getElementsByTagName('data').item(0);
% get text from child
data = elem.getFirstChild.getTextContent
% get attribute named att
att = elem.getAttribute('att')