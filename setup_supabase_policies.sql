-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- This sets up public access policies for the 'pdfs' storage bucket

-- Drop existing policies if any (safe to run multiple times)
DROP POLICY IF EXISTS "Allow public uploads"  ON storage.objects;
DROP POLICY IF EXISTS "Allow public reads"    ON storage.objects;
DROP POLICY IF EXISTS "Allow public deletes"  ON storage.objects;
DROP POLICY IF EXISTS "Allow public updates"  ON storage.objects;

-- Allow anyone to upload files to the pdfs bucket
CREATE POLICY "Allow public uploads"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'pdfs');

-- Allow anyone to read/download files from the pdfs bucket
CREATE POLICY "Allow public reads"
ON storage.objects FOR SELECT
USING (bucket_id = 'pdfs');

-- Allow anyone to delete files from the pdfs bucket
CREATE POLICY "Allow public deletes"
ON storage.objects FOR DELETE
USING (bucket_id = 'pdfs');

-- Allow anyone to update files in the pdfs bucket
CREATE POLICY "Allow public updates"
ON storage.objects FOR UPDATE
USING (bucket_id = 'pdfs');
