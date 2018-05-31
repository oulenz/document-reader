# document-reader

This is a class that can be used to read a document from a photo, given that the document is a filled out copy of a known template. The document is first identified and rescaled. Then certain sections of the document are labeled using pretrained neural networks.

The user needs to provide one or more template images, information on which parts of the documents to label, and the neural networks.

It may be convenient to extend this class, to add domain specific code.