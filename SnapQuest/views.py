import os
from django.shortcuts import render
from query_search import Image_search
import update_image_list
from DESM_query_serach import DESM_search
from positional_search import Pos_query_search
from django.conf import settings
from django.core.files.storage import FileSystemStorage

def landing_page(request):
    return render(request, 'landing.html')

def search_results(request):
    results = []
    submitted = False
    #Use this to variable to call the specific algo
    print(request.GET.get('search_type'))
    search_type = request.GET.get('search_type')
    if request.GET.get('query'):
        query = request.GET['query']
        if search_type == 'naive':
            results = Image_search(query)
        elif search_type == 'desm':
            results = DESM_search(query)
        elif search_type == 'positional':
            results = Pos_query_search(query)
        else:
            results = Image_search(query)
        submitted = True
    
    context = {
        'results': results,
        'query': query, 
        'submitted': submitted,
        'search_type': search_type
    }

    return render(request, 'landing.html', context)

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage(os.path.join(settings.STATICFILES_DIRS[0], 'new_images'))
        filename = fs.save(myfile.name, myfile)
        #return render(request, 'landing.html')
    update_image_list.refresh_index()
    return render(request, 'landing.html')